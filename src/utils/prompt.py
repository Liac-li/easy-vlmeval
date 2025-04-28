"""
提示词工具模块
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_prompt_template(task_name: str) -> Dict[str, str]:
    """
    加载提示词模板
    
    Args:
        task_name: 任务名称
        
    Returns:
        提示词模板字典
    """
    prompt_path = Path("configs/prompts/default.yaml")
    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词模板文件不存在: {prompt_path}")
        
    with open(prompt_path, "r") as f:
        prompts = yaml.safe_load(f)
        
    if task_name not in prompts["prompts"]:
        raise KeyError(f"未找到任务 {task_name} 的提示词模板")
        
    return prompts["prompts"][task_name] 