import json
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

import pandas as pd
import json
import numpy as np
import argparse
import os

import sys
import time

module_path = os.path.abspath("../..")
if module_path not in sys.path:
    sys.path.append(module_path)


# 1. BERT 모델 및 토크나이저 로드 (논문 표준: bert-base-uncased)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()

def get_knowledge_embedding(description:str, concepts:str):
    text = f"Problem Description: {description}\nKnowledge Concepts: {concepts}"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
        
    # ECKT와 같은 연구에서는 보통 전체 문맥을 담고 있는 [CLS] 토큰의 벡터를 사용하거나
    # 모든 토큰의 평균(Mean Pooling)을 사용합니다. 여기서는 [CLS]를 사용합니다.
    embeddings = outputs.last_hidden_state[:, 0, :].squeeze() # [768]
    return embeddings.cpu().numpy()

import src.make_logger as make_logger
from src.dataset_config import *
from src.utils import *
from src.misc import *

from exps.ECKT.extract_knowledge import extract_eckt_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="../../dataset/CodeWorkout")
    parser.add_argument("--result_dir", type=str, default="../../results")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print(args)

    save_dir = os.path.join(args.result_dir, "data")
    logger = make_logger.make("save_code_ke", save_dir=save_dir)
    logger.info(f"start")
    st_save_embed = time.time()

    emb_path = os.path.join(save_dir, 'code_knowledge_emb.tc')
    if os.path.isfile(emb_path):
        code_emb_dict = torch.load(emb_path, weights_only=False)
        code_emb_dict = {str(k): v for k, v in code_emb_dict.items()}
        torch.save(code_emb_dict, emb_path)
        exit(-1)

    code_path = os.path.join(args.dataset_dir, "LinkTables", "CodeStates.csv")
    code_df = pd.read_csv(code_path)
    logger.info(f"code_df.shape:{code_df.shape}")

    unique_code_id_dict = dict()
    code_duplicated = 0

    code_emb_dict = dict()

    code_knowledges = dict()

    total_processing = len(code_df)
    processed = 0
    retried = 0
    for idx, row in tqdm(code_df.iterrows(), total=total_processing):
        code_id = str(row["CodeStateID"])
        code = row["Code"]        
        
        if code in unique_code_id_dict:
            preprocessed_code_id = unique_code_id_dict[code]
            code_emb_dict[code_id] = code_emb_dict[preprocessed_code_id]
            code_duplicated += 1
        else:
            try:
                description, concepts, is_retry = extract_eckt_features(code)
            except Exception as e:
                logger.info(f"{processed}/{total_processing}: exception occurred. {code_id}")
                make_logger.write_exception_log(logger, e)
                exit(-1)

            if description is None:
                logger.info(f"{processed}/{total_processing}: failed extract eckt features. {code_id}")
                exit(-1)
                
            if is_retry:
                logger.info(f"{processed}/{total_processing}: retried while extracting eckt features. {code_id}")
                retried += 1
            
            code_knowledges[code_id] = {
                "desc":description, 
                "concepts": concepts
            }

            # BERT 벡터 추출
            emb = get_knowledge_embedding(description, concepts)

            code_emb_dict[code_id] = emb
            unique_code_id_dict[code] = code_id

        processed += 1
        if processed % 100 == 0 or processed==total_processing:
            logger.info(f"processed:{processed}/{total_processing}")
            logger.info(f"code duplicated: {code_duplicated}")
            logger.info(f"retried while extracting: {retried}")
            time.sleep(1)

        if processed % 1000 == 0:
            with open(os.path.join(save_dir, f"knowledge.json"), "w") as f:
                json.dump(code_knowledges, f, indent=3)

    logger.info(f"processed:{processed}")
    logger.info(f"code duplicated: {code_duplicated}")
    logger.info(f"retried while extracting: {retried}")

    torch.save(code_emb_dict, emb_path)

    with open(os.path.join(save_dir, f"knowledge.json"), "w") as f:
        json.dump(code_knowledges, f, indent=3)

    logger.info(f"end. elapse: {format_elapsed_time(time.time() - st_save_embed)}")

