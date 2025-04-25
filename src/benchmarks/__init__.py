"""
数据集模块，用于管理和下载基准测试数据集
"""

# 导出数据集类
from .mathvista import MathVistaDataset
from .mmmath import MMMathDataset

# 支持的数据集映射
supported_datasets = {
    'mathvista': MathVistaDataset,
    'mmmath': MMMathDataset,
}

__all__ = ['MathVistaDataset', 'MMMathDataset']
