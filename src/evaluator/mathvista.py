import argparse
import logging
import os
import re
from typing import List, Dict, Optional, Callable, Any, Union
import json
import pandas as pd
from pathlib import Path
from tasks.openai_runner import OpenAIRunner

from utils.logger import setup_logger

logger = setup_logger(__name__)


# pids = 852,  104,  824,  506,  540
demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Hint: Please answer the question requiring an integer answer and provide the final value, e.g., 1, 2, 3, at the end.
Question: Which number is missing?

Model response: The number missing in the sequence is 14.

Extracted answer: 14

Hint: Please answer the question requiring a floating-point number with one decimal place and provide the final value, e.g., 1.2, 1.3, 1.4, at the end.
Question: What is the fraction of females facing the camera?

Model response: The fraction of females facing the camera is 0.6, which means that six out of ten females in the group are facing the camera.

Extracted answer: 0.6

Hint: Please answer the question requiring a floating-point number with two decimal places and provide the final value, e.g., 1.23, 1.34, 1.45, at the end.
Question: How much money does Luca need to buy a sour apple candy and a butterscotch candy? (Unit: $)

Model response: Luca needs $1.45 to buy a sour apple candy and a butterscotch candy.

Extracted answer: 1.45

Hint: Please answer the question requiring a Python list as an answer and provide the final list, e.g., [1, 2, 3], [1.2, 1.3, 1.4], at the end.
Question: Between which two years does the line  graph saw its maximum peak?

Model response: The line graph saw its maximum peak between 2007 and 2008.

Extracted answer: [2007, 2008]

Hint: Please answer the question and provide the correct option letter, e.g., A, B, C, D, at the end.
Question: What fraction of the shape is blue?\nChoices:\n(A) 3/11\n(B) 8/11\n(C) 6/11\n(D) 3/5

Model response: The correct answer is (B) 8/11.

Extracted answer: B
"""

def verify_extraction(extraction):
    extraction = extraction.strip()
    if extraction == "" or extraction == None:
        return False
    return True


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def extract_answer(llm_gen: Callable[[str], str], response: str, problem: Dict[str, Any], quick_extract: bool = False) -> str:
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""

    if question_type == 'multi_choice' and response in choices:
        return response

    if answer_type == "integer":
        try:
            extraction = int(response)
            return str(extraction)
        except Exception as e:
            pass

    if answer_type == "float":
        try:
            extraction = str(float(response))
            return extraction
        except Exception as e:
            pass

    # quick extraction
    if quick_extract:
        logging.info("Quickly extracting answer...")
        # The answer is "text". -> "text"
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                extraction = result.group(1)
                return extraction
        except Exception as e:
            pass

    # general extraction
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        extraction = llm_gen(full_prompt)
        if extraction is None:
            return ""
        return extraction
    except Exception as e:
        logging.info(f"Error in extracting answer for problem: {pid} with response: {response}")
        logging.info(e)

    return ""


def get_most_similar(prediction, choices):
    """使用编辑距离确定哪个选项与给定预测最相似"""
    distances = []
    for choice in choices:
        # 计算简单的字符匹配距离
        if not isinstance(prediction, str):
            try:
                prediction = str(prediction)
            except:
                prediction = ""
        if not isinstance(choice, str):
            try:
                choice = str(choice)
            except:
                choice = ""
        distance = sum(1 for a, b in zip(prediction, choice) if a != b)
        distances.append(distance)
    ind = distances.index(min(distances))
    return choices[ind]

def normalize_extracted_answer(extraction, choices, question_type, answer_type, precision, ignore_empty_extractions=False):
    """标准化提取的答案以匹配答案类型"""
    if question_type == 'multi_choice':
        if isinstance(extraction, str):
            extraction = extraction.strip()
        else:
            try:
                extraction = str(extraction)
            except Exception:
                extraction = ""

        if ignore_empty_extractions and not extraction:
            return None

        letter = re.findall(r'\(([a-zA-Z])\)', extraction)
        if len(letter) > 0:
            extraction = letter[0].upper()

        sequential_characters = [chr(ord('A') + i) for i in range(len(choices))]

        if extraction in sequential_characters:
            option_index = sequential_characters.index(extraction)
            normalized_extraction = choices[option_index]
        else:
            normalized_extraction = get_most_similar(extraction, choices)
        assert normalized_extraction in choices

    elif answer_type == 'integer':
        try:
            normalized_extraction = str(int(float(extraction)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'float':
        try:
            normalized_extraction = str(round(float(extraction), int(precision)))
        except Exception:
            normalized_extraction = None

    elif answer_type == 'list':
        try:
            normalized_extraction = str(extraction)
        except Exception:
            normalized_extraction = None

    return normalized_extraction

def safe_equal(prediction, answer):
    """安全地检查预测是否等于答案"""
    try:
        if prediction == answer:
            return True
        return False
    except Exception as e:
        logging.info(e)
        return False

def get_acc_with_condition(res_pd, key, value):
    """计算特定条件下的准确率"""
    if key == 'skills':
        total_pd = res_pd[res_pd[key].apply(lambda x: value in x)]
    else:
        total_pd = res_pd[res_pd[key] == value]

    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = len(correct_pd) / len(total_pd) if len(total_pd) > 0 else 0

    return len(correct_pd), len(total_pd), acc

def calculate_scores(results_dict, ignore_empty_extractions=False):
    """计算总体和细粒度的评分"""
    # 计算总体准确率
    total = len(results_dict)
    correct = 0
    for pid, problem in results_dict.items():
        extraction = problem['extraction']
        choices = problem['choices']
        question_type = problem['question_type']
        answer_type = problem['answer_type']
        precision = problem['precision']
        answer = problem['answer']

        prediction = normalize_extracted_answer(
            extraction,
            choices,
            question_type,
            answer_type,
            precision,
            ignore_empty_extractions=ignore_empty_extractions
        )

        true_false = safe_equal(prediction, answer)
        problem['prediction'] = prediction
        problem['true_false'] = true_false

        if true_false:
            correct += 1

    accuracy = correct / total if total > 0 else 0
    scores = {"average": {"accuracy": accuracy, "correct": correct, "total": total}}

    # 计算细粒度准确率
    results_df = pd.DataFrame(results_dict).T
    
    # 合并 metadata
    for pid in results_dict:
        if 'metadata' in results_dict[pid]:
            results_dict[pid].update(results_dict[pid].pop('metadata'))

    target_keys = [
        'question_type',
        'answer_type',
        'language',
        'source',
        'category',
        'task',
        'context',
        'grade',
        'skills',
    ]

    for key in target_keys:
        if key == 'skills':
            values = []
            for i in range(len(results_df)):
                if key in results_df.iloc[i] and isinstance(results_df[key][i], list):
                    values += results_df[key][i]
            values = list(set(values))
        else:
            values = results_df[key].unique() if key in results_df else []

        scores[key] = {}
        for value in values:
            correct, total, acc = get_acc_with_condition(results_df, key, value)
            if total > 0:
                scores[key][value] = {"accuracy": acc, "correct": correct, "total": total}

        scores[key] = dict(sorted(scores[key].items(), key=lambda item: float(item[1]['accuracy']), reverse=True))

    return scores, results_dict

class MathVistaEvaluator:

    def __init__(self, results_file_path: Optional[str] = None, results: Optional[List[Dict[str, Any]]] = None):

        if results_file_path is None and results is None:
            raise ValueError("results_file_path or results must be provided")

        if results is not None:
            self.results = results
        else:
            assert results_file_path is not None
            with open(results_file_path, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
        
        self.model = OpenAIRunner(model_name="openai")
                
    def model_gen(self, user_prompt: str) -> str:

        result = self.model.generate(prompt=user_prompt)
        if result is None:
            return ""
        return result

    def extract_answer(self, result: Dict[str, Any]) -> str:

        extraction = extract_answer(self.model_gen, result['response'], result['origin_data'], quick_extract=True)
        return extraction

    def evaluate(self):
        """评估模型结果并返回评分"""
        extracted_results = {}
        logger.info(f"Evaluating {len(self.results)} results")
        logger.info(f"Extracting answers...")

        from tqdm import tqdm
        for result in tqdm(self.results, desc="Extracting answers", ncols=100):
            extraction = self.extract_answer(result)
            tmp = result.copy()
            tmp['extraction'] = extraction
            extracted_results[tmp['origin_data']['pid']] = tmp
        

        cal_usinig_results = {}
        for pid, result in extracted_results.items():
            tmp = result.copy()
            tmp.update(result['origin_data'])
            tmp.pop('origin_data')
            cal_usinig_results[pid] = tmp

        logger.info(f"Calculating scores...")
        scores, updated_results = calculate_scores(cal_usinig_results)
        
        # 使用json格式化输出分数，保持一致性
        logger.info("Final Evaluation Results:")
        logger.info(json.dumps(scores, indent=2))

        # post process, update the 'prediction' and 'true_false' in the extracted_results
        for pid, result in extracted_results.items():
            result['prediction'] = updated_results[pid]['prediction']
            result['true_false'] = updated_results[pid]['true_false']

        return extracted_results
            
if __name__ == "__main__":
    # export PYTHONPATH=$PYTHONPATH:/home/nfs06/liyt/easy-vlmeval/src
    evaluator = MathVistaEvaluator(results_file_path="/home/nfs05/yancy/easy-vllm/easy-vlmeval/output/mavis-grpo-90/mathvista_testmini_vllm_results_20250429_1042%.json")

    results = evaluator.evaluate()

    save_dir = Path('outputs/qwen2.5-vl-grpo-90') 
    save_path = save_dir / 'mathvista_testmini_vllm_results_evaluated.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)