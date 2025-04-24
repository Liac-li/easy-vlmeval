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

from utils.config import load_config
from utils.logger import setup_logger
from openai import OpenAI

logger = setup_logger(__name__)


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

class APIRunner:
    api_url_base = "http://localhost:7788/v1"
    api_key = "api-lyt-78fd9"
    
    def __init__(self, model_config: Dict, model_name: str, sys_prompt: str):

        if model_name is None or not os.path.exists(model_name):
            raise ValueError("Model name (local path) is required")

        self.sys_prompt = sys_prompt
        self.api_url_base = f"http://{model_config['host']}:{model_config['port']}/v1"
        self.api_key = model_config['api_key']
        logger.info(f"API URL: {self.api_url_base}")
        logger.info(f"API Key: {self.api_key}")
        self.model_name = model_name

        self.client = OpenAI(base_url=self.api_url_base, api_key=self.api_key)

    def _encode_image(self, image_path: str) -> str:
        """将图片转换为base64编码"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _prepare_image_message(self, image_path: str) -> Dict:
        """准备包含图片的消息格式"""

        base64_image = self._encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

    def _prepare_image_message_local_file(self, image_path: str) -> Dict:
        """准备包含图片的消息格式"""

        return {
            "type": "image_url",
            "image_url": {
                "url": image_path 
            }
        }


    def generate(self, query: Union[str, List], image_paths: List[str] = None, return_all: bool = False):
        """
        生成回复，支持文本和图片输入
        Args:
            query: 字符串或消息列表
            image_paths: 图片路径列表
            return_all: 是否返回完整响应
        """
        # 准备消息
        if isinstance(query, str):
            messages = [{"role": "system", "content": self.sys_prompt}]
            # 如果有图片，添加到消息中
            if image_paths:
                content = [{"type": "text", "text": query}]
                for img_path in image_paths:
                    content.append(self._prepare_image_message_local_file(img_path))
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": query})
        else:
            messages = query

        # 创建完成
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
        
        # Print the response
        response = completion.choices[0].message
        if return_all:
            return response
        else:
            return response.content

    def generate_with_file(self, query: str, image_paths: List[str] = None):
        """使用文件方式生成回复"""
        messages = [{"role": "system", "content": self.sys_prompt}]
        
        if image_paths:
            content = [{"type": "text", "text": query}]
            for img_path in image_paths:
                content.append(self._prepare_image_message_file(img_path))
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": query})

        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=2048,
        )
            
        return completion.choices[0].message.content

def process_batch(args):
    """处理单个数据样本的函数"""
    model, sample, prompt_template = args
        # 准备输入
    query = prompt_template.format(query=sample['query']) if prompt_template else sample['query']
        
    # 如果有图片，添加图片路径
    if 'image' in sample and sample['image']['path']:
        response = model.generate(query, image_paths=[sample['image']['path']])
    else:
        response = model.generate(query)
    
    return {
        'id': sample['id'],
        'query': sample['query'],
        'image_path': sample['image']['path'],
        'response': response,
        'ground_truth': {
            'answer': sample['answer'],
            'choices': sample['choices']
        },
        'success': True,
        'error': None
    }

def main(args, config):
    logger.info(f"Starting run with config path: {args.config}")

    # load dataset
    dataset_name, dataset_split = args.dataset.split('@')
    if dataset_name not in config['datasets']['config']:
        raise ValueError(f"Dataset {dataset_name} does not exist")
    
    dataset_config = config['datasets']['config'][dataset_name]
    dataset_class = supported_datasets[dataset_name]
    dataset = dataset_class(split=dataset_split, data_dir=dataset_config['local_dir'])[:20] #!
    logger.info(f"Dataset {dataset_name} loaded with size: {len(dataset)}")

    # 获取prompt模板
    prompt_config = config['prompts']['config'][args.prompt]
    sys_prompt = prompt_config['system']
    prompt_template = prompt_config['user']
    logger.info(f"Using prompt template: {prompt_template}")

    model = APIRunner(config['models']['config'][args.model], args.model_path, sys_prompt)

    # 存储所有结果
    all_results = []
    
    # 使用线程池进行并行处理
    for s_idx in range(0, len(dataset), args.num_workers):
        e_idx = min(s_idx + args.num_workers, len(dataset))
        samples = dataset[s_idx:e_idx]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # 准备任务列表
            tasks = [(model, sample, prompt_template) for sample in samples]
            
            # 提交所有任务并使用tqdm显示进度
            futures = list(tqdm(
                executor.map(process_batch, tasks),
                total=len(tasks),
                desc="Processing samples"
            ))
            
            # 收集结果（已经按顺序）
            all_results.extend(futures)

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
        # TODO: 实现评估逻辑
        pass


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(args, config)



