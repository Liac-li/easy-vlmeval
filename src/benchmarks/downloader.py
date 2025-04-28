from datasets import load_dataset, concatenate_datasets , get_dataset_config_names
import os
import json
import pandas as pd
from datasets import Image as HFImage
from PIL import Image

class DatasetDownloader:
    def __init__(self, dataset_name, subset: list=None, cache_dir=None):
        """
            从hf下载数据集
            dataset_name: 
            subset: 指定下载哪些大类,默认为所有
            cache_dir: 缓存路径
        """
        self.dataset_name = dataset_name
        self.subset = subset if subset else get_dataset_config_names(dataset_name)
        self.cache_dir = cache_dir
        self.dataset = None

    def download(self, split="validation"):
        self.split = split
        print(f"🔍 Loading dataset: {self.dataset_name}, split: {split}")
        if self.subset == None:
            self.dataset = load_dataset(
                self.dataset_name,
                name = self.subset,
                split = split,
                cache_dir=self.cache_dir
            )
            print(f"✅ Loaded {len(self.dataset)} samples.")
        
        else:
            sub_dataset_list = []
            for subject in self.subset:
                print(f"🔵 处理 config: {subject}")
                sub_dataset = load_dataset(
                    self.dataset_name, 
                    name = subject, 
                    split = split,
                    cache_dir=self.cache_dir)
                sub_dataset_list.append(sub_dataset)
                self.dataset = concatenate_datasets(sub_dataset_list)

            print(f"✅ Loaded {len(self.dataset)} samples.")

        return self.dataset

    def save_to_disk(self, output_dir, file_format="json"):
        """
            output_dir: 保存数据的文件夹 
                保存的路径--- output_dir/dataname/{split}.{file_format}
        """
        if self.dataset is None:
            raise ValueError("Dataset not downloaded yet. Call `download()` first.")
        
        output_dir += "/" + self.dataset_name.split("/")[-1]
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.split}.{file_format}")

        print(f"💾 Saving dataset to {output_path}")

        if file_format == "json":
            images_dir = os.path.join(output_dir, self.split)
            os.makedirs(images_dir , exist_ok = True)

            image_columns = [col for col, feature in self.dataset.features.items() if isinstance(feature, HFImage)]

            data_to_save = []
            for idx, sample in enumerate(self.dataset):
                sample_data = sample.copy()

                for image_col in image_columns:
                    img = sample_data.pop(image_col, None)
                    if img is not None:
                        img_filename = f"{image_col}_{idx}.png"  # 支持多个图片列
                        img_path = os.path.join(images_dir, img_filename)

                        if isinstance(img, Image.Image):
                            img.save(img_path)
                        else:
                            img = Image.fromarray(img)
                            img.save(img_path)

                        sample_data[f"{image_col}_path"] = os.path.relpath(img_path, output_dir)

                data_to_save.append(sample_data)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, ensure_ascii=False, indent=2)

        elif file_format == "parquet":
            self.dataset.to_parquet(output_path)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

        print(f"✅ Saved to {output_path}")

if __name__ == "__main__":
    downloader = DatasetDownloader("MMMU/MMMU")
    downloader.download(split="validation")
    downloader.save_to_disk(output_dir="data")