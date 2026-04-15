import os
import sys
import shutil

import numpy as np
import time

import tempfile
from pathlib import Path
from ray import tune, train
from ray.tune import Checkpoint

module_path = os.path.abspath("../../")
if module_path not in sys.path:
	sys.path.append(module_path)

from dataloader import get_train_data_loader
from evaluation import *

from eckt_model import ECKT

import src.make_logger as make_logger
from src.misc import *

def train(hp_config, **kwargs):
    import torch
    import torch.optim as optim

    hp_config_key_values = dict()
    for k, v in hp_config.items():
        if k=="learning_rate":
            k = "lr"
        elif k=="dropout":
            k = "do"
        elif k=="emb_size":
            k = "emb"
        elif k=="max_seq_len":
            k = "msl"
        hp_config_key_values[k] = v
    
    hp_config_str_list = []
    for key, value in hp_config_key_values.items():
        if key == "do":
            value = f"{value:.2f}"

        hp_config_str_list.append(f"{key}_{value}")

    is_hp_tuning = kwargs.get("is_hp_tuning", True)

    seed = kwargs["seed"]
    fold = kwargs["fold"]
    num_epochs = kwargs["num_epochs"]
    save_dir = kwargs["save_dir"]

    save_model_path = kwargs.get("save_model_path")

    logger_name = f"fold_{fold}_"
    logger_name += "-".join(hp_config_str_list)

    logger = make_logger.make(logger_name, time_filename=False, save_dir=save_dir)
    logger.info(f"start. fold:{fold}")

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info(f"cuda_visible_devices: [{cuda_visible_devices}]. Using device: {device} for this trial.")

    setup_seed(seed)
    
    if is_hp_tuning:
        hp_config_key_values['lr'] = f"{hp_config_key_values['lr']:.10f}"
        if 'dropout' in hp_config_key_values:
            hp_config_key_values['dropout'] = f"{hp_config_key_values['dropout']:.2f}"
        log_prefix = "-".join([f"{key}_{value}" for key, value in hp_config_key_values.items()])
    else:
        log_prefix = "train"

    logger.info(f"hp_config: {hp_config}")

    st = time.time()

    es_delta = kwargs.get("es_delta", 0)
    es_patience = kwargs.get("es_patience")

    train_data_dir = kwargs["train_data_dir"]

    input_info = kwargs["input_info"]
    model_config = kwargs["model_config"]

    model_type = model_config["model_type"]
    logger.info(f"model_type: {model_type}")

    n_problems = model_config["n_problems"]

    max_seq_len = hp_config.get("max_seq_len")
    if max_seq_len is not None:
        model_config["max_seq_len"] = max_seq_len

    hp_emb_size = hp_config.get("emb_size")
    if hp_emb_size is not None:
        model_config["emb_size"] = hp_emb_size

    hp_dropout = hp_config.get("dropout")
    if hp_dropout is not None:
        model_config["dropout"] = hp_dropout

    rnn_stack = hp_config.get("rnn_stack", 1)
    learning_rate = hp_config["learning_rate"]

    model_config['rnn_num_layers'] = rnn_stack

    # logger.info(f"model_config:"+json.dumps(model_config, indent=3))

    train_msg = f"problems: {n_problems}. bs:{model_config['batch_size']}"
    logger.info(f"{train_msg}. start")

    try:
        cache_train_data_path = os.path.join(os.path.abspath(train_data_dir), f"train_data_fold{fold}.npy")
        cache_valid_data_path = os.path.join(os.path.abspath(train_data_dir), f"valid_data_fold{fold}.npy")

        logger.info(f"load train data: {cache_train_data_path}")
        train_data = np.load(cache_train_data_path, allow_pickle=True)

        logger.info(f"load valid data: {cache_valid_data_path}")
        valid_data = np.load(cache_valid_data_path, allow_pickle=True)

        logger.info(f"data shape: {train_data.shape}, {valid_data.shape}")
        train_loader, test_loader = get_train_data_loader(model_config['batch_size'],
                                                          train_data,
                                                          valid_data,
                                                          model_config["max_seq_len"],
                                                          logger)
    except Exception as e:
        make_logger.write_exception_log(logger, e, "train")
        exit(-1)

    logger.info(f"create data loader.end elapse:{format_elapsed_time(time.time()-st)}")

    logger.info(f"init model")
    model = ECKT(
        model_type=model_type,
        input_info=input_info,
        input_dim=n_problems * 2,
        hidden_dim=model_config['hidden'],
        rnn_num_layers=model_config['rnn_num_layers'],
        output_dim=n_problems,
        emb_size=model_config["emb_size"],
        dropout=model_config["dropout"],
        max_code_len=model_config['max_code_len'],
        num_problems=n_problems,
        problem_rank_dim=model_config['problem_rank_dim'],
        code_knowledge_dim=model_config['code_knowledge_dim'],
        att_dim_org=model_config["att_dim_org"],
        has_w0=model_config["has_w0"],
        logger=logger
        )
    model.to(device)

    gpu_per_trial = kwargs.get("gpu_per_trial", 1)
    if gpu_per_trial > 1:
        model = nn.DataParallel(model)
        logger.info(f"using DataParallel: gpus:{torch.cuda.device_count()}. gpu_per_trial:{gpu_per_trial}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = lossFunc(n_problems, model_config["max_seq_len"], device)

    trial_cfg = {"hp_config":hp_config,
                 "model_config": model_config,
                 "input_info": input_info
                 }

    torch.cuda.empty_cache()

    logger.info(f"train start")

    best_val_auc = 0
    epochs_since_best = 0

    best_checkpoint_data = None
    for epoch in range(1, num_epochs+1):
        logger.info(f"{log_prefix}. epoch {epoch}/{num_epochs}. start")

        try:
            epoch_st = time.time()
            loss = train_epoch(
                model, 
                train_loader, 
                optimizer,
                loss_func, 
                n_problems, 
                device)

            logger.info(f"\t eval")
            performance = evaluate(model, test_loader, loss_func, device)

            val_auc = performance[0]
            val_acc = performance[-1]

            last_since_best = epochs_since_best
            if val_auc > best_val_auc + es_delta:
                epochs_since_best = 0
            else:
                epochs_since_best += 1

            best_val_auc = max(best_val_auc, val_auc)

            metrics = {"val_auc":val_auc, 
                       "best_val_auc": best_val_auc,
                       "val_acc":val_acc, 
                       "train_loss":float(loss)}

            checkpoint_data = {
                "trial_cfg": trial_cfg,
                'epoch': epoch,
                "epochs_since_best": epochs_since_best,
                'metrics': metrics,
            }

            logger.info(f"\t{log_prefix}. epoch:{epoch} end. auc:{val_auc:.5f}, best:{best_val_auc:.5f}"
                        f", since_best:{epochs_since_best}, last_since_best:{last_since_best}")

            epoch_elapse = format_elapse_from(epoch_st)

            torch.cuda.empty_cache()

            report_st = time.time()
            if is_hp_tuning:
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    model_ckpt_path = Path(checkpoint_dir) / "model_tune.ckpt"
                    torch.save(checkpoint_data, model_ckpt_path)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)
                    tune.report(
                        metrics,
                        checkpoint=checkpoint,
                    )        
            else:
                if epochs_since_best == 0:
                    best_checkpoint_data = checkpoint_data
                    best_checkpoint_data.update({
                        'model_state': model.state_dict() if not isinstance(model, nn.DataParallel) else model.module.state_dict(),
                        'optimizer_state': optimizer.state_dict()
                    })

            if epoch % 10 == 1:
                logger.info(f"\t{log_prefix}. epoch:{epoch}. "
                            f"elapse. epoch: {epoch_elapse}"
                            f", report: {format_elapse_from(report_st)}"
                            f", total: {format_elapse_from(st)}")

            if es_patience is not None and epochs_since_best >= es_patience:
                logger.info(f"epoch:{epoch}. Early stopped. best_epoch={epochs_since_best}, best_auc={best_val_auc}")
                break

        except Exception as e:
            make_logger.write_exception_log(logger, e, "train")
            exit(-1)

    if not is_hp_tuning:
        try:
            torch.save(best_checkpoint_data, save_model_path)
        except Exception as e:
            make_logger.write_exception_log(logger, e, "train")
            exit(-1)

    logger.info(f"{log_prefix}. completed! elapse: total {format_elapsed_time(time.time() - st)}")
