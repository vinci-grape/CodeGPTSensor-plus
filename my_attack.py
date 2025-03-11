import os
import sys
import signal

from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    T5ForConditionalGeneration,
    T5EncoderModel,
    T5Config,
    T5Tokenizer,
)
import torch
import argparse
import logging
import json
import random
import pandas as pd
import numpy as np

from model import CodeGPTSensor
from mist import MIST
from utils import Example

logger = logging.getLogger(__name__)

# path of the saved huggingface models
UNIXCODER_PATH = "/data2/xiaodanxu/huggingface/unixcoder-base-nine"
CODEBERT_PATH = "/data2/xiaodanxu/huggingface/codebert-base"

import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Function call timed out")

def load_victim_model(model_type, saved_victim_path, device, n_gpu, args):
    if model_type == "codegptsensor":
        config = RobertaConfig.from_pretrained(UNIXCODER_PATH)
        tokenizer = RobertaTokenizer.from_pretrained(UNIXCODER_PATH)
        model = RobertaModel.from_pretrained(UNIXCODER_PATH)
        model = CodeGPTSensor(model, config, tokenizer, args)
        # model_to_load = model.module if hasattr(model, 'module') else model
        # model_to_load.load_state_dict(torch.load(saved_victim_path))
        model.load_state_dict(torch.load(saved_victim_path))
        logger.info("Loaded victim model from {}.".format(saved_victim_path))
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
    elif model_type == "codebert":
        config = RobertaConfig.from_pretrained(CODEBERT_PATH)
        tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_PATH)
        model = RobertaModel.from_pretrained(CODEBERT_PATH)
        model = CodeGPTSensor(model, config, tokenizer, args)
        # model_to_load = model.module if hasattr(model, 'module') else model
        # model_to_load.load_state_dict(torch.load(saved_victim_path))
        model.load_state_dict(torch.load(saved_victim_path))
        logger.info("Loaded victim model from {}.".format(saved_victim_path))
        model.to(device)
        if n_gpu > 1:
            model = torch.nn.DataParallel(model)
    else:
        raise ValueError(
            "Invalid model type: {}".format(
                model_type
            )
        )
    return model, tokenizer

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    # Load parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_file",
        required=True,
        type=str,
        help="The path to the attacked data file",
    )
    parser.add_argument(
        "--saved_victim_model_path",
        required=True,
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        default="codebert",
        type=str,
        help="The type of the victim model",
    )
    parser.add_argument(
        "--attack_numbers",
        default=100,
        type=int,
        help="The number of examples to attack",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="The random seed for the attack",
    )
    parser.add_argument(
        "--result_store_path",
        default="attack_results.csv",
        type=str,
        help="The path to store the attack results",
    )

    parser.add_argument(
        "--language",
        default="",
        type=str,
        help="The programming language, python or java.",
    )

    parser.add_argument("--index", nargs='+',
        help="Optional input sequence length after tokenization.")

    args = parser.parse_args()

    # Set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() 
    args.device = device
    logger.info("Args %s", args)
    logger.info("device: %s, n_gpu: %s", args.device, args.n_gpu)

    # Set seed
    set_seed(args.seed)

    # load victim model
    model, tokenizer = load_victim_model(args.model_type, args.saved_victim_model_path, args.device, args.n_gpu, args)

    # load T5 model
    # codet5_path = "Salesforce/codet5-base"
    codet5_path = "/data2/xiaodanxu/huggingface/codet5-base"
    t5_tokenizer = RobertaTokenizer.from_pretrained(codet5_path)
    t5_config = T5Config.from_pretrained(codet5_path)
    t5_emb_model = T5EncoderModel.from_pretrained(codet5_path, config=t5_config)
    t5_emb_model.to(device)  # this model is for similarity calculation

    # [start_idx, end_idx)
    start_idx = int(args.index[0])
    end_idx = int(args.index[1])
    
    # 加载测试数据
    finished_idx = []
    try:
        finished = pd.read_json(args.result_store_path, lines=True)
        finished_idx = list(finished['idx'])
        logger.info("Finished idxs: %s", str(finished_idx[:5] + ["..."] + finished_idx[-5:]))
        logger.info("Skip %s finished examples. Start attack!", str(len(finished_idx)))
    except:
        logger.info("Start attack!")

    examples = []
    with open(args.data_file) as f:
        for i, line in enumerate(f):
            if i < start_idx or i >= end_idx or (i in finished_idx):
                continue
            js = json.loads(line.strip())
            examples.append(
                Example(idx=js['idx'], source=js['code'].strip(), target=js['label'], index=js['index'], code_tokens=js['code_tokens'], identifiers=js['identifiers'], replace_dict=js['replace_dict'])
            )

    logger.info("Number of examples: %s, idx: %s~%s", len(examples), start_idx, end_idx)

    # # sample the attacked examples
    # examples = random.sample(examples, min(args.attack_numbers, len(examples)))
    # logger.info("Numer of sampled examples={}".format(len(examples)))

    POP_SIZE = 30
    ITERATION_LIMIT = 50
    
    # Initialize the attacker
    attacker = MIST(
        args,
        model,
        tokenizer,
        t5_tokenizer,
        t5_emb_model,
        pop_size=POP_SIZE,
        iteration_limit=ITERATION_LIMIT,
    )

    logger.info('>>> set POP_SIZE = %s, ITERATION_LIMIT = %s', POP_SIZE, ITERATION_LIMIT)

    TIME_OUT = 60 # 1分钟
    logger.info('>>> set TIME_OUT = %s', TIME_OUT)

    total_cnt = 0
    success_cnt = 0
    for i, example in enumerate(examples, start=1):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(TIME_OUT)
        try:
            is_success, time_cost = attacker.attack(example)
            signal.alarm(0)
        except TimeoutException as e:
            signal.alarm(0)
            logger.info('idx = %s, Time Out!', str(example.idx))
            is_success, time_cost = -5, TIME_OUT / 60     # 有时候程序不知道为啥会卡死，于是设置了超时返回失败，时间设为3分钟
            res = {
                "idx": example.idx, "index": example.index,
                "code": example.source, "variables": None, "true_label": example.target, 
                "orig_pred_label": None, "orig_prob": None, "adv_code": None, "adv_variables": None, 
                "adv_pred_label": None, "adv_true_label_confidence": None,
                "is_success": is_success, "nb_queries": None, "time_cost": time_cost,
                "F1": None, "F2": None, "F3": None, "evolution_round": None, "max_iteration": None
            }
            with open(args.result_store_path, "a") as wf:
                wf.write(json.dumps(res)+'\n')
        
        if is_success == 1 or is_success == -1: # the original example is attacked, i.e. correctly predicted and has the identifiers to be perturbed
            total_cnt += 1
        if is_success == 1:
            success_cnt += 1        
        logger.info("[%s/%s] idx = %s, is_success = %s, time_cost = %s min, success_rate = %s", str(i), str(len(examples)), str(example.idx), str(is_success), str(round(time_cost, 2)), round(success_cnt / (total_cnt + 1e-16), 3))


    logger.info(
        "*****All Finished, total_num={}, success_num={}, success_rate={}*****".format(
            total_cnt,
            success_cnt,
            float(success_cnt / (total_cnt + 1e-16)),
        )
    )

if __name__ == "__main__":
    main()
