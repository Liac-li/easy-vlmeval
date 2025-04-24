"""
MathVista数据集加载器
"""

from typing import Literal, Optional, Dict, Any
from datasets import load_dataset, Dataset
from pathlib import Path
import json
import base64
from PIL import Image
import io

class MathVistaDataset:
    def __init__(
        self,
        split: Literal["test", "testmini"] = "testmini",
        data_dir: str = "benchmarks/MathVista"
    ):
        """
        初始化MathVista数据集
        
        Args:
            split: 数据集划分，可选 "test" 或 "testmini"
            data_dir: 数据目录路径
        """
        self.split = split
        self.dataset_path = data_dir
        self.data_dir = Path(self.dataset_path) / "data"
        self.image_dir = Path(self.dataset_path) / "images"
        self.dataset = self._load_dataset()
        
    def _load_dataset(self) -> Dataset:
        """
        从本地加载parquet数据集
        
        Returns:
            Dataset: 加载的数据集
        """
        if self.split == "testmini":
            # testmini只有一个parquet文件
            pattern = "testmini-*.parquet"
            parquet_files = list(self.data_dir.glob(pattern))
            if len(parquet_files) == 0:
                raise FileNotFoundError(f"找不到{pattern}文件")
            parquet_path = parquet_files[0]
        elif self.split == "test":
            # test有两个parquet文件
            pattern = "test-*.parquet"
            parquet_files = list(self.data_dir.glob(pattern))
            if len(parquet_files) == 0:
                raise FileNotFoundError(f"找不到{pattern}文件")
            parquet_paths = sorted(parquet_files)
        else:
            raise ValueError(f"不支持的split: {self.split}")
            
        if not parquet_path.exists() if self.split == "testmini" else not all(p.exists() for p in parquet_paths):
            raise FileNotFoundError(f"Parquet文件不存在: {parquet_path if self.split == 'testmini' else parquet_paths}")
            
        # 加载parquet文件
        if self.split == "testmini":
            return load_dataset("parquet", data_files=str(parquet_path))["train"]
        else:
            return load_dataset("parquet", data_files=[str(p) for p in parquet_paths])["train"]
            
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.dataset)
        
    def __getitem__(self, idx: int) -> dict:
        """获取单个样本"""
        ret = self.convert_to_inference_format(idx)
        return ret
        
    def convert_to_inference_format(self, idx: int) -> Dict[str, Any]:
        """
        将数据集样本转换为推理格式
        
        Args:
            idx: 样本索引
            
        Returns:
            Dict: 转换后的数据格式
        """
        sample = self.dataset[idx]
        
        # 获取图像数据
        
        # 将PIL图像转换为base64
        # buffered = io.BytesIO()
        # image.save(buffered, format="PNG")
        # img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # 构建转换后的数据
        converted_data = {
            "id": sample['pid'],
            "query": sample['query'],
            "answer": sample['answer'],
            "choices": sample['choices'] if sample['choices'] else None,
            "image": {
                "path": str(self.image_dir / sample['image']),
                "bytes": sample['decoded_image']
            },
            "origin_data": sample  # 保留原始数据
        }
        
        return converted_data
        
    @property
    def features(self) -> dict:
        """返回数据集特征"""
        return self.dataset.features


if __name__ == "__main__":
    # for example in shot_examples:
    #     print("----------------------------------------")
    #     print("\nQuestion:", example['question'])
    #     if "choices" in example:
    #         print("\nChoices:", example['choices'])
    #     print("\nCaption:", example['caption'])
    #     if "ocr" in example:
    #         print("\nOCR:", example['ocr'])
    #     print("\nSolution:", example['solution'])
    #     print("\nCode:", example['code'])

    #     # execute code
    #     exec(example['code'])
        
    dataset = MathVistaDataset(split="test")
    print(dataset[0])

   