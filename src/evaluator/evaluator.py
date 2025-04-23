"""
评估器的主要实现
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from ..utils.config import load_config
from ..utils.prompt import load_prompt_template
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

class Evaluator:
    def __init__(self, config_path: str = "configs/models/default.yaml"):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.vllm_url = f"http://{self.config['vllm']['host']}:{self.config['vllm']['port']}/v1/chat/completions"
        
    def evaluate(self, task_name: str, data_path: str) -> List[Dict[str, Any]]:
        """
        评估指定任务
        
        Args:
            task_name: 任务名称
            data_path: 数据路径
            
        Returns:
            评估结果列表
        """
        logger.info(f"开始评估任务: {task_name}")
        
        # 加载提示词模板
        prompt_template = load_prompt_template(task_name)
        
        # 加载数据集
        with open(data_path, "r") as f:
            dataset = json.load(f)
            
        results = []
        for item in dataset:
            # 构建提示词
            prompt = self._build_prompt(item, prompt_template)
            
            # 调用模型
            response = self._call_model(prompt)
            
            # 记录结果
            results.append({
                "id": item["id"],
                "prediction": response,
                "ground_truth": item["answer"]
            })
            
        return results
        
    def _build_prompt(self, item: Dict[str, Any], template: Dict[str, str]) -> str:
        """
        构建提示词
        
        Args:
            item: 数据项
            template: 提示词模板
            
        Returns:
            构建好的提示词
        """
        return template["user"].format(**item)
        
    def _call_model(self, prompt: str) -> str:
        """
        调用模型
        
        Args:
            prompt: 提示词
            
        Returns:
            模型响应
        """
        # TODO: 实现模型调用逻辑
        pass
        
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        保存评估结果
        
        Args:
            results: 评估结果
            output_path: 输出路径
        """
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"评估结果已保存到: {output_path}") 