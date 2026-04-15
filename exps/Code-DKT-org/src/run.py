
import os
import random
import pandas as pd

import torch

import torch.optim as optim
import numpy as np

from dataloader import get_data_loader
import evaluation
import warnings
warnings.filterwarnings("ignore")

from c2vRNNModel import c2vRNNModel
from config import Config

import sys
module_path = os.path.abspath("../../..")
if module_path not in sys.path:
    sys.path.append(module_path)
import src.make_logger as make_logger

def setup_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import tqdm

def main(gpu_id, org_att_dim, dkt, sort_by_time):
    config = Config()

    setup_seed(0)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    logger_name = f"run-{'dkt' if dkt else 'codedkt'}-{'org_att_dim' if org_att_dim else 'valid_att_dim'}-sort_by_time_{sort_by_time}"
    logger = make_logger.make(logger_name, save_dir="../results/log")
    logger.info(f"org_att_dim:{org_att_dim}. gpu_id-{gpu_id}. sort_by_time:{sort_by_time} start")

    overall_column_valus = []
    columns = []
    for assignment_id in sorted(os.listdir("../data")):
        data_dir = os.path.join("../data", assignment_id)
        if not os.path.isdir(data_dir):
            continue
        
        if sort_by_time:
            feat_dir = os.path.join("../data_sort_by_time", assignment_id)
        else:
            feat_dir = data_dir

        logger.info(f"assignment_id:{assignment_id}. start")

        performance_list = []
        scores_list = []
        first_scores_list = []
        first_total_scores_list = []

        for fold in range(10):
            logger.info(f"----{fold}-th run----")
            train_loader, test_loader = get_data_loader(data_dir, feat_dir, config.bs, config.questions, config.length, fold, dkt)
            node_count, path_count = np.load(f"{data_dir}/np_counts.npy")
            if not dkt:
                logger.info(f"node_count, path_count = {node_count}, {path_count}")

            model = c2vRNNModel(config.questions * 2,
                                config.hidden,
                                config.layers,
                                config.questions,
                                node_count, path_count, device,
                                org_att_dim,
                                dkt) 

            optimizer = optim.Adam(model.parameters(), lr=config.lr)
            loss_func = evaluation.lossFunc(config.questions, config.length, device)
            for epoch in tqdm.tqdm(range(config.epochs)):
                model, optimizer = evaluation.train_epoch(model, train_loader, optimizer,
                                                loss_func, config, device)
            first_total_scores, first_scores, scores, performance = evaluation.test_epoch(
                model, test_loader, loss_func, device, epoch, config, fold)
            first_total_scores_list.append(first_total_scores)
            scores_list.append(scores)
            first_scores_list.append(first_scores)
            performance_list.append(performance)
            logger.info(f"----{fold}-th auc:{performance}")


        performance_mean = np.mean(performance_list,axis=0)
        performance_mean = [f"{float(x):.4f}" for x in performance_mean]
        performance_std = np.std(performance_list,axis=0)
        performance_std = [f"{float(x):.4f}" for x in performance_std]
        performance_mean_std = f"{performance_mean[0]},{performance_std[0]}"

        logger.info(f"Average scores of all attempts:{performance_mean_std}")

        overall_column_valus.append(performance_mean_std)
        columns.append(assignment_id)
        logger.info(f"assignment_id:{assignment_id}. end \n")


    if dkt:
        if not sort_by_time:
            exp_type = "-"
        else:
            exp_type = "*TAlign"
    else:
        if not sort_by_time:
            exp_type = "-"
        else:
            if org_att_dim:
                exp_type = "TAlign"
            else:
                exp_type = "*CRect"

    rows = [["DKT" if dkt else "CodeDKT", f"${exp_type}$"] + overall_column_valus]
    columns = ["Model", "Setting"] + columns

    result_save_path = f"../results/results-{'dkt' if dkt else 'codedkt'}-{exp_type}.csv"
    df = pd.DataFrame(rows, columns = columns)
    df.to_csv(result_save_path, index=False)

    from exps.results_to_latex_table import save_latex_table
    result_save_path = f"../results/results-latex-{'dkt' if dkt else 'codedkt'}-{exp_type}.txt"
    save_latex_table(df, result_save_path, vert_line=False, std_float_point="0.2f")
    logger.info(f"org_att_dim:{org_att_dim}.end")

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--valid_att_dim", action="store_true")
    parser.add_argument("--dkt", action="store_true")
    parser.add_argument('--sort_by_time', action="store_true")

    args = parser.parse_args()

    main(args.gpu_id, not args.valid_att_dim, args.dkt, args.sort_by_time)
