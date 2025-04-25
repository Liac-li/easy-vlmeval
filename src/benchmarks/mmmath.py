
from pathlib import Path
import json 
from torch.utils.data import Dataset
import os
from typing import Literal


class MMMathDataset(Dataset):
    def __init__(
            self, 
            split: Literal["train"] = "train", 
            data_dir: str = "data/MM_Math"):

        # self.split = split
        self.data_dir = Path(os.path.abspath(data_dir))
        self.datafile = self.data_dir / "MM_Math" /"MM_Math.jsonl"
        self.dataset = self._load_dataset()
        
        self.dataset = self.convert_to_inference_format()

    def _load_dataset(self):

        dataset = []
        with open(self.datafile, "r") as f:
            for line in f.readlines():
                data = json.loads(line)
                dataset.append(data)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def convert_to_inference_format(self):

        converted_datas = []
        from tqdm import tqdm
        for i in tqdm(range(len(self.dataset)), desc="转换数据格式", unit="样本"):
            sample = self.dataset[i]
            # 删除decoded_image字段
            
            # 构建转换后的数据
            converted_data = {
                "id": i,
                "query": sample['question'],
                "answer": sample['solution'],
                "choices": sample.get('choices', None),
                "image": {
                    "path": str(self.data_dir / "MM_Math" / "MM_Math" / sample['file_name']),
                    "bytes": None
                },  
                "origin_data": sample
            }
            converted_datas.append(converted_data)

        return converted_datas


if __name__ == "__main__":
    dataset = MMMathDataset()
    print(len(dataset))
    print(dataset[0])
