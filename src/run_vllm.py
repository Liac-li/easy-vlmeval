import argparse
import os
import base64
from typing import Union, List, Dict
from PIL import Image
from io import BytesIO
import concurrent.futures
from tqdm import tqdm
from datetime import datetime
import json

from benchmarks import supported_datasets
from evaluator import supported_evaluators
from utils.config import load_config
from utils.logger import setup_logger
from tasks import QwenVLRunner

logger = setup_logger(__name__)

# export VLLM_WORKER_MULTIPROC_METHOD=spawn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config", help="Path to config file")
    parser.add_argument("--model", type=str, default="vllm", help="Model name")
    parser.add_argument("--model-path", type=str, default="", help="Model Checkpoint Path")
    parser.add_argument("--gpu-num", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--dataset", type=str, default="", help="Dataset name") # name@split, all lower
    parser.add_argument("--prompt", type=str, default="", help="Prompt template")
    parser.add_argument("--output-dir", type=str, default="", help="Output directory")
    parser.add_argument("--num-workers", type=int, default=4, help='Number of workers')
    
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate")
    parser.add_argument("--reuse-saved", action="store_true", help="Whether to reuse saved results")

    return parser.parse_args()

def main(args, config):
    logger.info(f"Starting run with config path: {args.config}")

    # load dataset
    dataset_name, dataset_split = args.dataset.split('@')
    if dataset_name not in config['datasets']['config']:
        raise ValueError(f"Dataset {dataset_name} does not exist")
    
    dataset_config = config['datasets']['config'][dataset_name]
    dataset_class = supported_datasets[dataset_name]
    dataset = dataset_class(split=dataset_split, data_dir=dataset_config['local_dir']) #!
    logger.info(f"Dataset {dataset_name} loaded with size: {len(dataset)}")

    # 获取prompt模板
    prompt_config = config['prompts']['config'][args.prompt]
    sys_prompt = prompt_config['system']
    prompt_template = prompt_config['user']
    logger.info(f"Using prompt template: {prompt_template}")

    model = QwenVLRunner(args.model_path, args.gpu_num)

    # 存储所有结果
    all_results = []
    
    for s_idx in range(0, len(dataset), args.num_workers):
        e_idx = min(s_idx + args.num_workers, len(dataset))
        samples = dataset[s_idx:e_idx]
        
        queries = [prompt_template.format(query=sample['query']) 
                   for sample in samples]
        images_path = [sample['image']['path'] for sample in samples]
        
        batch_results = model.generate(queries, images_path)

        for sample, result in zip(samples, batch_results):
            all_results.append({
                'id': sample['id'],
                'query': sample['query'],
                'image': sample['image']['path'],
                'response': result,
                'origin_data': sample['origin_data']
            })

    # 保存结果
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(
            args.output_dir,
            f"{dataset_name}_{dataset_split}_{args.model}_results_{timestamp}.json"
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_path}")

    # 如果需要评估
    if args.eval:
        evaluator_class = supported_evaluators[dataset_name]
        evaluator = evaluator_class(results_file_path=output_path, results=all_results)
        evaluated_results = evaluator.evaluate()

        # 保存评价过的结果
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(
                args.output_dir,
                f"{dataset_name}_{dataset_split}_{args.model}_evaluated_results_{timestamp}.json"
            )
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluated_results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to {output_path}")




if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(args, config)



