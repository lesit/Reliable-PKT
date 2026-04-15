
import os
import time
import numpy as np
import pandas as pd

import argparse

import sys
module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)

import src.make_logger as make_logger
from src.misc import *

import env_config
from exps.ray_tune import hp_tune_fold
from exps.model_folds_cv import train_cv_on_data
# from exps.model_folds_cv_random_init_model import train_cv_on_data

import eckt_ray_trainable
import eckt_eval

import make_cached_data

from ray import tune

# grid search
# tune_config_grid={
#     "search_space": {
#         "emb_size": tune.grid_search([50, 100, 150, 300, 350]),
#         "dropout": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
#         "learning_rate": tune.grid_search([0.00005, 0.0001, 0.0005, 0.001])
#     },
#     "num_samples": 1
# }

tune_config_grid={
    "search_space": {
        "emb_size": tune.grid_search([50, 100, 150, 300, 350]),
        "dropout": tune.grid_search([0.1, 0.2, 0.3, 0.4, 0.5]),
        "learning_rate": tune.grid_search([0.00005, 0.0001, 0.0005])
    },
    "num_samples": 1
}

tune_config_opt={
    "search_space":{
        "emb_size": tune.choice([50, 100, 150, 300, 350]),
        "dropout": tune.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        "learning_rate": tune.loguniform(5e-5, 5e-4)
    },
    "num_samples": 100
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type')

    parser.add_argument('--has_w0', action="store_true")
    parser.add_argument('--all_grid', action="store_true")
    parser.add_argument('--att_dim_valid', action="store_true")

    parser.add_argument('--msl', type=int)

    parser.add_argument("--batch_size", type=int, default=None)

    parser.add_argument("--seed", default=128)
    parser.add_argument("--num_epochs", default=100)
    parser.add_argument("--scheduler_grace_period", default=15)
    parser.add_argument("--es_delta", default=1e-4)
    parser.add_argument("--es_patience", default=10)

    parser.add_argument("--num_samples", type=int, default=None)

    parser.add_argument("--result_dir", type=str, default="../../results")

    parser.add_argument("--compare_dir", type=str, default=None)

    args = parser.parse_args()
    arg_params = vars(args)

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    seed = args.seed
    num_epochs = args.num_epochs

    if args.batch_size is None:
        batch_size = env_config.ModelConfig().batch_size
    else:
        batch_size = args.batch_size

    if args.all_grid:
        tune_config = tune_config_grid
    else:
        tune_config = tune_config_opt

    search_space = tune_config["search_space"]
    num_samples = tune_config["num_samples"]

    if args.all_grid:
        num_samples = 1

        search_space_str_list = [f"{key}_{value['grid_search']}" for key, value in search_space.items()]
    else:
        if args.num_samples is not None:
            num_samples = args.num_samples

        search_space_str_list = [f"{key}_{value.domain_str}" for key, value in search_space.items()]

    search_space_name = "-".join(search_space_str_list)
    search_space_name = search_space_name.replace(' ', '')

    result_dir = args.result_dir

    opt_str = "train"
    if batch_size != 32:
        opt_str += f"-bs{batch_size}"

    tune_desc = ("hp_grid_" if args.all_grid else "hp_bayesian_") + search_space_name + f"-msl_{args.msl}"

    result_save_root_dir =  os.path.join(result_dir, tune_desc)

    exp_save_dir = os.path.join(result_save_root_dir, args.model_type)
    if args.model_type != "dkt":
        opt_str += "-att_dim_valid" if args.att_dim_valid else "-att_dim_org"

        exp_save_dir = os.path.join(exp_save_dir, "has_w0" if args.has_w0 else "no_w0")

    tune_save_dir = os.path.join(exp_save_dir, f"{opt_str}-1-hp_tune")
    cv_save_dir = os.path.join(exp_save_dir, f"{opt_str}-2-cv_folds")

    logger_name = f"hp_tune"
    logger = make_logger.make(logger_name, save_dir=tune_save_dir)
    logger.info(f"\hp_tune start. args:\n{arg_params}")

    logger.info(f"search space: {search_space_name}")
    logger.info(f"num_samples:{num_samples}")
    logger.info(f"batch_size:{batch_size}")

    program_st = time.time()

    data_dir = os.path.join(result_dir, "data", "assignments")

    fold = 0

    if args.model_type == "dkt":
        search_space.pop("emb_size")
        search_space.pop("dropout")

    assignment_hp_config_list = []

    if args.model_type == "dkt":
        exp_type = "TAlign"
    else:
        if not args.att_dim_valid:
            exp_type = "TAlign"
        else:
            exp_type = "CRect"
            if args.has_w0:
                exp_type += "+"

    eval_list = []
    for assignment in sorted(os.listdir(data_dir)):
        data_assignment_dir = os.path.join(data_dir, assignment)

        problems_d = np.load(os.path.join(data_assignment_dir, "problems.npy"), allow_pickle=True).item()
        n_problems = len(problems_d)

        cached_data_info = make_cached_data.get_cached_data_info(True, data_assignment_dir)
        if cached_data_info is None:
            logger.info("no cached train data")
            exit(-1)

        train_data_dir, input_info, n_folds = cached_data_info
        logger.info(f"input_info:{input_info}")

        model_config = env_config.ModelConfig()
        model_config.batch_size = batch_size

        model_config.model_type = args.model_type
        model_config.att_dim_org = not args.att_dim_valid
        model_config.n_problems = n_problems
        model_config.has_w0 = args.has_w0
        model_config.max_seq_len = args.msl

        train_data_dir = os.path.abspath(train_data_dir)
        hp_tune_save_dir = os.path.abspath(os.path.join(tune_save_dir, assignment))

        logger.info(f"train_hp_tune_cv. assignment:{assignment}. fold:{fold}. save dir:{hp_tune_save_dir}. tune start")
        tune_st = time.time()

        try:
            hp_tune_fold(
                10000,
                seed,
                num_epochs,
                search_space, 
                num_samples,
                args.scheduler_grace_period,
                args.es_delta,
                args.es_patience,
                fold,
                eckt_ray_trainable.train,
                hp_tune_save_dir,
                logger,
                train_data_dir=train_data_dir,
                model_config=model_config.__dict__,
                input_info=input_info)
        except Exception as e:
            make_logger.write_exception_log(logger, e, f"hp_tune_fold. fold:{fold}")
            exit(-1)

        logger.info(f"train_hp_tune_cv. assignment:{assignment}. fold:{fold}. save dir:{hp_tune_save_dir}. tune end.  elapse:{format_elapse_from(tune_st)}")

        cv_assignment_dir = os.path.join(cv_save_dir, assignment)

        import json
        trial_cfg_path = os.path.join(hp_tune_save_dir, "best_checkpoint.ckpt_trial_cfg.json")
        with open(trial_cfg_path, "r") as f:
            trial_cfg = json.load(f)
        hp_config = trial_cfg["hp_config"]
        model_config = trial_cfg["model_config"]

        logger.info(f"train_hp_tune_cv. cv assignment:{assignment}. save dir:{cv_assignment_dir} start")
        cv_st = time.time()

        eval_batch_size = 128
        try:
            eval_df = train_cv_on_data(
                eckt_ray_trainable.train,
                eckt_eval.eval_fold,
                eval_batch_size,
                seed,
                num_epochs,
                args.es_delta,
                args.es_patience,
                hp_config,
                train_data_dir, 
                n_folds, 
                cv_assignment_dir,
                logger,
                input_info = input_info,
                model_config = model_config,
                data_unit_name = "assignment",
                data_unit_type = assignment
                )
            
            logger.info(f"train_hp_tune_cv. cv assignment:{assignment}. save dir:{cv_assignment_dir} end. elapse:{format_elapse_from(cv_st)}")
        except Exception as e:
            make_logger.write_exception_log(logger, e, f"train_cv_on_data.")
            exit(-1)

        if eval_df is None:
            break

        row = eval_df[eval_df["fold"] == "mean_std"]
        row = row.iloc[0]
        eval_list.append([row['assignment'], row['testauc']])

        assignment_hp_config_list.append([args.model_type, exp_type, assignment] + list(hp_config.values()) + [row['train_epoch']])

    from exps.results_to_latex_table import save_latex_table

    columns = ['Model', 'Setting'] + [eval[0] for eval in eval_list]
    
    if args.model_type=="dkt":
        model_disp = "*DKT"
    else:
        if args.model_type=="codedkt":
            model_disp = "CodeDKT"
        else:
            model_disp = "ECKT"
        if args.att_dim_valid:
            model_disp = "*"+model_disp

    rows = [model_disp, "$"+exp_type+"$"]

    rows += [eval[1] for eval in eval_list]
    rows = [rows]

    save_path = os.path.join(result_save_root_dir, f"result-{args.model_type}-{exp_type}.csv")
    result_df = pd.DataFrame(rows, columns=columns)
    result_df.to_csv(save_path, index=False)

    from pathlib import Path

    def make_total_result_df(saved_dir):
        df_list = []
        target_dir = Path(saved_dir)
        files = sorted(target_dir.iterdir(), key=lambda f: f.stat().st_mtime)
        for file in files:
            if not file.name.endswith(".csv"):
                continue
            csv_path = file.resolve()
            result_df = pd.read_csv(csv_path)
            df_list.append(result_df)
        return pd.concat(df_list, ignore_index=True)

    total_df = make_total_result_df(result_save_root_dir)
    latex_save_path = os.path.join(result_save_root_dir, f"total_latex_table_tune-{tune_desc}.txt")
    save_latex_table(total_df, latex_save_path, vert_line=False, mean_float_point="0.4f", std_float_point="0.2f")

    if args.compare_dir is None:
        df_for_latex = total_df
    else:
        cmp_df = make_total_result_df(args.compare_dir)

        rows = []
        for index, row in total_df.iterrows():
            rows.append(list(row.values))

            model_type = row["Model"]
            setting = row["Setting"]
            cmp_row = cmp_df.query("Model == @model_type and Setting == @setting")
            if len(cmp_row) == 0:
                continue
            cmp_row = cmp_row.iloc[0]
            delta_cols = ["", "\small $\Delta$"]
            for col_name in total_df.columns[2:]:
                cur_v = float(row[col_name].split(',')[0])
                cmp_v = float(cmp_row[col_name].split(',')[0])
                diff = cur_v - cmp_v

                if diff > 0:
                    str_diff = "$\\textbf{"+f"{diff:+.4f}"+"}$"
                else:
                    str_diff = "${"+f"{diff:+.4f}"+"}$"
                delta_cols.append(str_diff)
            rows.append(delta_cols)
            rows.append(["\\hline"] + [None for x in range(len(total_df.columns)-1)])

        df_for_latex = pd.DataFrame(rows, columns=total_df.columns)

        latex_save_path = os.path.join(result_save_root_dir, f"total_delta_latex_table_tune-{tune_desc}.txt")
        save_latex_table(df_for_latex, latex_save_path, vert_line=False, mean_float_point="0.4f", std_float_point="0.2f")

    columns = ['Model', 'Setting', 'assignment'] + list(hp_config.keys()) + ['epoch']
    hp_total_df = pd.DataFrame(assignment_hp_config_list, columns=columns)
    hp_total_df.to_csv(os.path.join(cv_save_dir, f"hyper_parameters_{tune_desc}-{args.model_type}.csv"), index=False)
    logger.info(f"completed!!!. total elapse:{format_elapsed_time(time.time()-program_st)}")
