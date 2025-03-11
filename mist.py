import sys

# sys.path.append("../")
sys.path.append("./parser")

import time
import json
import logging
import numpy as np
import pandas as pd
from utils import Individual, Population
import logging
from utils import Example

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

class MIST:
    def __init__(
        self, args, model, tokenizer, t5_tokenizer, t5_emb_model, pop_size, iteration_limit
    ):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.t5_tokenizer = t5_tokenizer
        self.t5_emb_model = t5_emb_model
        self.pop_size = pop_size
        self.iteration_limit = iteration_limit
        self.parent_pop = Population(self.pop_size)
        self.offspring_pop = Population(self.pop_size)
        self.mixed_pop = Population(self.pop_size * 2)

        self.do_crossover = True
        logger.info("do_crossover: %s", self.do_crossover)

    # def attack(self, idx, orig_code, label):
    def attack(self, item: Example):
        self.parent_pop.indi = []
        self.offspring_pop.indi = []

        orig_code = item.source
        idx = item.idx
        label = item.target
        index = item.index

        code_tokens = item.code_tokens
        identifiers = item.identifiers
        replace_dict = item.replace_dict

        # check whether the original code can be predicted correctly by victim model
        prob = self.model.predict(orig_code)
        orig_label = np.argmax(prob)
        orig_prob = np.max(prob)

        query_times = 0
        flag_success = False  # whether we have found the first adversarial example in our algorithm
        evolution_round = 0
        is_success = 0

        # 如果异常，就直接保存结果并return
        if orig_label != label:  # the original code is misclassified
            is_success = -4
        elif len(identifiers) == 0:  # no identifier in the code
            is_success = -3
        
        if is_success < -1:
            res = {
                "idx": idx, "index": index, "code": orig_code, 
                "variables": None, "true_label": label, 
                "orig_pred_label": int(orig_label), "orig_prob": float(orig_prob),
                "adv_code": None, "adv_variables": None,  "adv_pred_label": None, 
                "adv_true_label_confidence": None, "is_success": is_success, 
                "nb_queries": None, "time_cost": None,
                "F1": None, "F2": None, "F3": None, 
                "evolution_round": None, "max_iteration": None
            }
            with open(self.args.result_store_path, "a") as wf:
                wf.write(json.dumps(res)+'\n')
            return is_success, 0
        
        # Set max iteration
        self.max_iteration = min(5 * len(identifiers), self.iteration_limit)
        # self.max_iteration = 5 * len(identifiers)

        adv_example = {}
        
        # 开始attack!
        attack_start_time = time.time()
        try:
            # 初始化population
            for i in range(self.pop_size):
                self.parent_pop.indi.append(
                    Individual(orig_code, code_tokens, identifiers, replace_dict, orig_label, self.args.language)
                )
                self.offspring_pop.indi.append(
                    Individual(orig_code, code_tokens, identifiers, replace_dict, orig_label, self.args.language)
                )
                self.parent_pop.indi[i].mutation()
                self.mixed_pop.indi.append(
                    Individual(orig_code, code_tokens, identifiers, replace_dict, orig_label, self.args.language)
                )
                self.mixed_pop.indi.append(
                    Individual(orig_code, code_tokens, identifiers, replace_dict, orig_label, self.args.language)
                )

                self.parent_pop.indi[i].function_eval(
                    self.model, self.tokenizer, self.t5_emb_model, self.t5_tokenizer
                )
                query_times += 1            
                if self.parent_pop.indi[i].label_ != orig_label:
                    flag_success = True
                    adv_example = self.parent_pop.indi[i]
                    break
            
            # 开始evolution迭代
            if flag_success == False:
                for i in range(1, self.max_iteration + 1):
                    evolution_round = i
                    # crossover
                    if self.do_crossover:
                        # print('Iteration', i, 'crossover!', flush=True)
                        self.offspring_pop.crossover(self.parent_pop)
                    # mutation
                    # print('Iteration', i, 'mutation!', flush=True)
                    self.offspring_pop.mutation()
                    # evaluate the objectives of the offspring population
                    for j in range(self.pop_size):
                        self.offspring_pop.indi[j].function_eval(
                            self.model,
                            self.tokenizer,
                            self.t5_emb_model,
                            self.t5_tokenizer,
                        )
                        query_times += 1
                        if self.offspring_pop.indi[j].label_ != orig_label:
                            flag_success = True
                            adv_example = self.offspring_pop.indi[j]
                            break

                    if flag_success:
                        break

                    # environmental selection
                    self.parent_pop.environmental_selection(
                        self.offspring_pop, self.mixed_pop
                    )
            
            # 结束attack，保存结果
            time_cost = (time.time() - attack_start_time) / 60  # min
            
            # 再检查一下有没有成功的
            if flag_success == False:
                for indi in self.parent_pop.indi:
                    if indi.label_ != orig_label:
                        flag_success = True
                        adv_example = self.offspring_pop.indi[j]
                        break

            if flag_success == True:
                is_success = 1
            else:
                is_success = -1
                min_indi = None
                min_f1 = 9.9
                # 选择在true_label上的confidence最低的
                for j, indi in enumerate(self.parent_pop.indi):
                    if indi.obj_[0] < min_f1:
                        min_f1 = indi.obj_[0]
                        min_indi = indi
                adv_example = min_indi
            
            # 保存对抗样本，并return
            res = {
                "idx": idx, 
                "index": index,
                "code": orig_code, 
                "variables": identifiers, 
                "true_label": label, 
                "orig_pred_label": int(orig_label), 
                "orig_prob": float(orig_prob),
                "adv_code": " ".join(adv_example.tokens_),
                "adv_variables": adv_example.identifiers, 
                "adv_pred_label": int(adv_example.label_), 
                "adv_true_label_confidence": float(adv_example.obj_[0]),
                "is_success": is_success, 
                "nb_queries": query_times,
                "time_cost": time_cost,
                "F1": float(adv_example.obj_[0]),
                "F2": adv_example.obj_[1],
                "F3": adv_example.obj_[2],
                "evolution_round": evolution_round,
                "max_iteration": self.max_iteration
            }

            with open(self.args.result_store_path, "a") as wf:
                wf.write(json.dumps(res)+'\n')
            return is_success, time_cost
        except Exception as e:
            logger.error(e)
            is_success = -6   
            res = {
                "idx": idx, "index": index, "code": orig_code, 
                "variables": None, "true_label": label, 
                "orig_pred_label": int(orig_label), "orig_prob": float(orig_prob),
                "adv_code": None, "adv_variables": None,  "adv_pred_label": None, 
                "adv_true_label_confidence": None, "is_success": is_success, 
                "nb_queries": None, "time_cost": None,
                "F1": None, "F2": None, "F3": None, 
                "evolution_round": None, "max_iteration": None
            }
            with open(self.args.result_store_path, "a") as wf:
                wf.write(json.dumps(res)+'\n')
            return is_success, 0
    