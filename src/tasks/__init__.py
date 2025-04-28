"""
任务模块，定义不同的评估任务
"""

from .qwen_runner import QwenVLRunner
from .openai_runner import OpenAIRunner

__all__ = ['QwenVLRunner', 'OpenAIRunner'] 