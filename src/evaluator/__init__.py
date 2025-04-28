"""
评估器模块，用于评估模型在基准测试上的表现
"""

from .mathvista import MathVistaEvaluator
from .mmmath import MMMathEvaluator
from .mmmu import MMMUEvaluator

supported_evaluators = {
    "mathvista": MathVistaEvaluator,
    "mmmath": MMMathEvaluator,
    "mmmu": MMMUEvaluator
}

__all__ = ['MathVistaEvaluator', 'MMMathEvaluator', 'MMMUEvaluator']
