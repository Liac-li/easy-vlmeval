"""
评估器模块，用于评估模型在基准测试上的表现
"""

from .mathvista import MathVistaEvaluator
from .mmmath import MMMathEvaluator

supported_evaluators = {
    "mathvista": MathVistaEvaluator,
    "mmmath": MMMathEvaluator
}

__all__ = ['MathVistaEvaluator', 'MMMathEvaluator']
