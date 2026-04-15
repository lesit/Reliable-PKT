import os
import numpy as np
import time
import torch

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
	sys.path.append(module_path)

import src.make_logger as make_logger
from src.misc import *

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:2'

from dataloader import get_test_data_loader
from eckt_model import ECKT
from evaluation import *

def eval_fold(batch_size,
              refined_data_dir, 
              checkpoint_path,
              logger,
              **kwargs):
    log_prefix = kwargs.get("log_prefix")
    logger.info(f"{log_prefix}. eval start")

    eval_fold_st = time.time()
    if not os.path.isfile(checkpoint_path):
        return None

    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_id = torch.cuda.current_device()
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    logger.info(f"cuda_visible_devices: [{cuda_visible_devices}]. gpu_id: {gpu_id}. Using device: {device} for this eval.")

    checkpoint_data = torch.load(checkpoint_path)
    trial_cfg = checkpoint_data["trial_cfg"]
    epoch = checkpoint_data["epoch"]
    model_state = checkpoint_data["model_state"]
    optimizer_state = checkpoint_data["optimizer_state"]

    model_config = trial_cfg["model_config"]
    input_info = trial_cfg["input_info"]

    model_type = model_config["model_type"]
    n_problems = model_config["n_problems"]
    max_seq_len = model_config["max_seq_len"]

    logger.info(f"model_type: {model_type}")

    cache_test_data_path = os.path.join(os.path.abspath(refined_data_dir), f"test_data.npy")
    test_data = np.load(cache_test_data_path, allow_pickle=True)

    test_loader = get_test_data_loader(batch_size=batch_size, 
                                       test_data=test_data, 
                                       max_seq_len=max_seq_len, 
                                       logger=logger)

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

    model.load_state_dict(model_state)

    loss_func = lossFunc(n_problems, max_seq_len, device)

    torch.cuda.empty_cache()

    performance = evaluate(model, test_loader, loss_func, device)

    testauc = performance[0]
    testacc = performance[-1]

    torch.cuda.empty_cache()

    dres = {"testauc": testauc, "testacc": testacc, "train_epoch": epoch}  

    logger.info(f"{log_prefix}. eval end. elapse: {format_elapse_from(eval_fold_st)}\n{dres}")
    return dres
