"""
评估器模块，用于评估模型在基准测试上的表现
"""

from .mathvista import MathVistaEvaluator

supported_evaluators = {
    "mathvista": MathVistaEvaluator
}

__all__ = ['MathVistaEvaluator']
