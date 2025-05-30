import argparse
import os
import base64
import pickle
from typing import Union, List, Dict, Any
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
    
    parser.add_argument('--save-step', type=int, default=100, help='每多少步保存一次结果')
    parser.add_argument("--eval", action="store_true", help="Whether to evaluate")
    parser.add_argument("--reuse-saved", type=str, default="", help="Whether to reuse saved results")

    return parser.parse_args()

def batch_auto_load_image(image_data: List[Dict[str, Any]]):
    """
        Auto load image from image_data
    """
    image_data_list = []
    has_multiple = False
    for data in image_data:
        image_data, is_multiple = auto_load_image(data)
        has_multiple = True if is_multiple else has_multiple 
        image_data_list.append(image_data)
    return image_data_list, has_multiple

def auto_load_image(image_data: Dict[str, Any]):
    """
        Supported image data keys:
        - path[str]: path to image
        - bytes[bytes]: bytes of image
        - item[Image.Image]: PIL image
        
        return: 
            [data, is_list]
    """
    image_path = image_data.get('path')
    if image_path is not None:
        if isinstance(image_path, str):
            return image_path, False
        elif isinstance(image_path, list):
            return image_path, True
    elif image_data.get('item') is not None:
        if isinstance(image_data['item'], Image.Image):
            return image_data['item'], False
        elif isinstance(image_data['item'], list):
            return image_data['item'], True
    else:
        raise ValueError(f"Unsupported image data type: {type(image_data)}")
    

def main(args, config):
    logger.info(f"Starting run with config path: {args.config}")

    # load dataset
    dataset_name, dataset_split = args.dataset.split('@')
    if dataset_name not in config['datasets']['config']:
        raise ValueError(f"Dataset {dataset_name} does not exist")
    
    dataset_config = config['datasets']['config'][dataset_name]
    dataset_class = supported_datasets[dataset_name]
    dataset = dataset_class(split=dataset_split, data_dir=dataset_config['local_dir'], reuse_saved=args.reuse_saved) #!
    dataset_max_image_per_prompt = dataset.max_image_per_prompt
    logger.info(f"Dataset {dataset_name} loaded with size: {len(dataset)}")
    logger.info(f"Dataset {dataset_name} max image per prompt: {dataset_max_image_per_prompt}")

    # 获取prompt模板
    prompt_config = config['prompts']['config'][args.prompt]
    sys_prompt = prompt_config['system']
    prompt_template = prompt_config['user']
    logger.info(f"Using prompt template: {prompt_template}")

    model = QwenVLRunner(args.model_path, args.gpu_num, max_image_per_prompt=dataset_max_image_per_prompt)

    # 存储所有结果
    all_results = []
    total = len(dataset)
    save_step = args.save_step
    
    with tqdm(total=(total//args.num_workers)+1, desc="推理进度") as pbar:
        for s_idx in range(0, total, args.num_workers):
            e_idx = min(s_idx + args.num_workers, total)
            samples = dataset[s_idx:e_idx]
            
            queries = [prompt_template.format(query=sample['query']) 
                       for sample in samples]
            images_data, has_multiple = batch_auto_load_image([sample['image'] for sample in samples])

            try:

                batch_results = model.generate(queries, images_data)

                for sample, result in zip(samples, batch_results):
                    all_results.append({
                        'id': sample['id'],
                        'query': sample['query'],
                        'image': sample['image']['path'],
                        'response': result,
                        'origin_data': sample['origin_data']
                    })
                pbar.update(1)

            except Exception as e:
                logger.error(f"Error generating results for batch {s_idx}-{e_idx}: {e}")
                continue

            # 每save_step步保存一次（覆盖同一个中间文件）
            if args.output_dir and len(all_results) % save_step == 0:
                os.makedirs(args.output_dir, exist_ok=True)
                os.chmod(args.output_dir, 0o777)
                step_output_path = os.path.join(
                    args.output_dir,
                    f"{dataset_name}_{dataset_split}_{args.model}_intermediate.pkl"
                )
                with open(step_output_path, 'wb') as f:
                    pickle.dump(all_results, f)
                logger.info(f"Step {len(all_results)}: Results saved to {step_output_path}")

    # 保存最终结果
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        # 确保目录有正确的权限
        os.chmod(args.output_dir, 0o777)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_path = os.path.join(
            args.output_dir,
            f"{dataset_name}_{dataset_split}_{args.model}_results_{timestamp}.pkl"
        )
        with open(output_path, 'wb') as f:
            pickle.dump(all_results, f)
        logger.info(f"Results saved to {output_path}")

    # 如果需要评估
    if args.eval:
        evaluator_class = supported_evaluators[dataset_name]
        evaluator = evaluator_class(results_file_path=output_path, results=all_results)
        evaluated_results = evaluator.evaluate()

        # 保存评价过的结果
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
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