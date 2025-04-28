"""
任务模块，定义不同的评估任务
"""

from .base import BaseTask
from .mmmu_task import MMMUTask
from .mathvista_task import MathVistaTask

__all__ = ['BaseTask', 'MMMUTask', 'MathVistaTask'] 