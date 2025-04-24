"""
数据集模块，用于管理和下载基准测试数据集
"""

from .mmmu import MMMUDataset
from .mathvista import MathVistaDataset
from .downloader import DatasetDownloader

supported_datasets = {
    'mathvista': MathVistaDataset,
}

__all__ = ['MMMUDataset', 'MathVistaDataset', 'DatasetDownloader']
