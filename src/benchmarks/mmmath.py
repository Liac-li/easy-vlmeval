
from pathlib import Path
import json 
from torch.utils.data import Dataset
import os
from typing import Literal


class MMMathDataset(Dataset):
    def __init__(
            self, 
            split: Literal["train"] = "train", 
            data_dir: str = "data/MM_Math",
            reuse_saved: str = None
        ):

        # self.split = split
        self.data_dir = Path(os.path.abspath(data_dir))
        self.datafile = self.data_dir / "MM_Math" /"MM_Math.jsonl"
        self.dataset = self._load_dataset()
        
        self.dataset = self.convert_to_inference_format()
        if reuse_saved:
            self.saved_results_path = Path(os.path.abspath(reuse_saved))
            self._clean_dataset()
            
    def _clean_dataset(self):
        """
        清理数据集, 删除已经推理过的样本
        """ 
        with open(self.saved_results_path, "r") as f:
            saved_results = json.load(f)
        saved_pids = [result['id'] for result in saved_results]
        
        # filter out the ids in saved
        self.dataset = [item for item in self.dataset if item['pid'] not in saved_pids]
            
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
