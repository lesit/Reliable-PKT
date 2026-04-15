import os
import json
import shutil
import torch

os.environ["RAY_DISABLE_METRICS"] = "1"

import ray
from ray import tune
from ray.tune.search.optuna import OptunaSearch 
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper, CombinedStopper, MaximumIterationStopper
from ray.tune.search import ConcurrencyLimiter, BasicVariantGenerator

from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger
from src.misc import *

def hp_tune_fold(
    min_gpu_per_trial,
    seed,
    num_epochs,
    search_space, 
    num_samples,
    scheduler_grace_period,
    es_delta,
    es_patience,
    fold,
    trainable,
    save_dir,
    logger,
    **kwargs):

    save_best_checkpoint_path = os.path.join(save_dir, "best_checkpoint.ckpt")
    save_best_hp_path = os.path.join(save_dir, "best_found.json")
    if os.path.isfile(save_best_checkpoint_path) and os.path.isfile(save_best_hp_path):
        logger.info(f"already done: {save_best_checkpoint_path}")
        return

    exp_name = f"train_fold_{fold}"
    logger.info(f"{exp_name}. start")
    fold_st = time.time()

    if ray.is_initialized():
        ray.shutdown()

    torch.cuda.empty_cache()

    grid_search = num_samples==1

    n_cpus = int(os.cpu_count() / 2)
    gpu_free_list = get_free_gpu_list(logger)
    n_gpus = len(gpu_free_list)

    logger.info(f"grid_search:{grid_search}, logical CPU cores: {n_cpus}, n_gpus: {n_gpus}")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(gpu_id) for gpu_id in range(n_gpus)])
    
    resources_per_trial = get_resources_per_trial(n_cpus, gpu_free_list, min_gpu_per_trial, logger)
    gpu_per_trial = resources_per_trial["gpu"]

    ray.init(num_cpus=n_cpus, 
             num_gpus=n_gpus,
             configure_logging=False,
    )
    logger.info(f"cluster_resources: {ray.cluster_resources()}")

    if grid_search:
        scheduler = None
        search_alg = None
    else:
        scheduler = ASHAScheduler(
            time_attr="training_iteration",
            metric="val_auc",
            mode="max",
            max_t=num_epochs,
            grace_period=scheduler_grace_period,
            reduction_factor=2
        )

        search_alg = OptunaSearch(
            metric="val_auc",
            mode="max",
            seed=seed,
        )

    stopper = EarlyStopper(
        metric="val_auc",
        mode="max",
        delta=es_delta,
        patience=es_patience,
    )

    trainable = tune.with_parameters(
        trainable,
        gpu_per_trial=gpu_per_trial,

        seed=seed,
        fold=fold,
        num_epochs=num_epochs,
        save_dir=save_dir,
        **kwargs
    )
    checkpoint_config = ray.tune.impl.config.CheckpointConfig(
                    checkpoint_score_attribute="val_auc", 
                    checkpoint_score_order="max" 
                )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources=resources_per_trial),
        param_space=search_space,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=num_samples
        ),
        run_config=tune.RunConfig(
            storage_path=save_dir,
            name="ray_tune",  # storage_path 아래에 name 폴더가 생김... 그래서. 중복됨.
            stop=stopper,
            checkpoint_config=checkpoint_config,
            log_to_file=os.path.join(save_dir, "ray_tune_main.log")
        )
    )

    try:
        result_grid = tuner.fit()
    except Exception as e:
        make_logger.write_exception_log(logger, e, f"{exp_name}. tuner.fit()")
        exit(-1)

    logger.info("finding best using best_val_auc")
    best_result = result_grid.get_best_result(
        metric="best_val_auc",
        mode="max",
        scope="all"
    )
    best_checkpoint = best_result.get_best_checkpoint(metric="val_auc", mode="max")
    save_best_checkpoint(exp_name, best_result, best_checkpoint, save_best_checkpoint_path, save_best_hp_path, logger)

    all_trials_df = result_grid.get_dataframe()
    all_trials_df.to_csv(os.path.join(save_dir, "all_trials_df.csv"), index=False)

    ray.shutdown()

    logger.info(f"{exp_name}. end. elapse:{format_elapsed_time(time.time()-fold_st)}\n save_best_checkpoint_path:{save_best_checkpoint_path}")

    ray_tune_saved_dir = os.path.join(save_dir, "ray_tune")
    if os.path.isdir(ray_tune_saved_dir):
        shutil.rmtree(ray_tune_saved_dir)
        logger.info(f"{exp_name}. rmtree:{ray_tune_saved_dir}")

from ray.tune.stopper import Stopper

class EarlyStopper(Stopper):
    def __init__(self, metric, mode, delta=1e-3, patience=10):
        self.metric = metric
        self.is_max_mode = mode == "max"
        self.delta = delta
        self.patience = patience
        self.best_scores = {}  
        self.counts = {}       

    def __call__(self, trial_id, result):
        current_score = result.get(self.metric)
        if current_score is None:
            return False

        if trial_id not in self.best_scores:
            self.best_scores[trial_id] = current_score
            self.counts[trial_id] = 0
            return False

        best_score = self.best_scores[trial_id]

        if self.is_max_mode and current_score > best_score + self.delta or\
            not self.is_max_mode and current_score < best_score - self.delta:
            self.best_scores[trial_id] = current_score
            self.counts[trial_id] = 0
        else:
            self.counts[trial_id] += 1

            if self.counts[trial_id] >= self.patience:
                print(f"\n[Early Stop] Trial {trial_id} stopped. Peak: {self.best_scores[trial_id]:.4f}, Current: {current_score:.4f}")
                return True

        return False

    def stop_all(self):
        return False

def save_best_checkpoint(exp_name, result, checkpoint:ray.train.Checkpoint, save_best_checkpoint_path, save_best_hp_path, logger):
    with checkpoint.as_directory() as checkpoint_dir:
        ckpt_path = Path(checkpoint_dir) / f"model_tune.ckpt"
        logger.info(f"saved best ckpt_path: {ckpt_path}")

        checkpoint_data = torch.load(ckpt_path)

    trial_cfg = checkpoint_data["trial_cfg"]
    epoch = checkpoint_data["epoch"]
    metrics = checkpoint_data["metrics"]

    hp_config = trial_cfg.get("hp_config")

    logger.info(f"{exp_name}. Best hyperparameters found were: {hp_config}")
    logger.info(f"{exp_name}. result metrics of best checkpoint: {metrics}")
    logger.info(f"{exp_name}. epoch of best checkpoint: {epoch}")

    try:
        with open(f"{save_best_checkpoint_path}_trial_cfg.json", "w") as fout:
            json.dump(trial_cfg, fout, ensure_ascii=False, indent=3)
    except Exception as e:
        make_logger.write_exception_log(logger, e, f"hp_tune_fold. json.dump({save_best_checkpoint_path}_trial_cfg.json)")

    try:
        shutil.copy2(ckpt_path, save_best_checkpoint_path)
    except Exception as e:
        make_logger.write_exception_log(logger, e, f"hp_tune_fold. shutil.copy2({ckpt_path}, {save_best_checkpoint_path})")
        exit(-1)

    training_iteration = result.metrics["training_iteration"]
    best_found = {
        "hp_config": hp_config,
        "metrics": metrics,
        "epoch": epoch,
        "training_iteration": training_iteration
    }
    with open(save_best_hp_path, "w") as fout:
        json.dump(best_found, fout, ensure_ascii=False, indent=3)

def get_free_gpu_list(logger):
    n_gpus = torch.cuda.device_count()

    gpu_free_list = []
    for i_gpu in range(n_gpus):
        with torch.cuda.device(i_gpu):
            free_mem, total_mem = torch.cuda.mem_get_info()
            free_mem /= (1024**2)
            total_mem /= (1024**2)
            logger.info(f"{i_gpu} gpu: {free_mem:.2f} MiB / {total_mem:.2f} MiB")
            gpu_free_list.append(free_mem)
    gpu_free_list = sorted(gpu_free_list)

    return gpu_free_list

def get_resources_per_trial(n_cpus, gpu_free_list, min_gpu_per_trial, logger):
    n_gpus = len(gpu_free_list)

    found_double = False
    if len(gpu_free_list)==1:
        found_double = True
        for free in gpu_free_list:
            if free < min_gpu_per_trial*2:
                found_double = False
                break

    found_less = False
    for free in gpu_free_list:
        if free < min_gpu_per_trial:
            found_less = True
            break

    if found_double:
        gpu_per_trial = 0.5
        n_trial = int(n_gpus / gpu_per_trial)
    elif found_less:
        gpu_per_trial = None
        accum = 0
        for i_gpu in range(n_gpus):
            accum += gpu_free_list[i_gpu]
            if accum >= min_gpu_per_trial:
                gpu_per_trial = i_gpu + 1
                break
        if gpu_per_trial is None:
            logger.info(f"not enough gpu memory")
            exit(-1)

        n_trial = 1
    else:
        gpu_per_trial = 1
        n_trial = n_gpus

    cpu_per_trial = min(int(n_cpus / n_trial), 8)  # cpu가 많을 수록 더 빠름. 4일때 34.7, 6일때 33, 8일때 32
    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpu_per_trial}

    logger.info(f"ray_tune. n_trial:{n_trial}, resources_per_trial:{resources_per_trial}")
    return resources_per_trial
