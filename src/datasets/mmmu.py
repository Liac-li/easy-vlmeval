
"""Utils for data load, save, and process (e.g., prompt construction)"""

import os
import json
import yaml
import ast
import re
from typing import List, Dict, Any
from PIL import Image
class MMMUDataset:
    def __init__(self,data_fold: str , split_type:str , file_type = ".json"):
        """
        初始化 MMMUDataset 类。
        :param data_path: 数据文件(Parquet)的本地路径。
        """
        self.dataname = "mmmu"
        self.data_fold = data_fold
        self.split_type = split_type

        self.data_path = os.path.join(self.data_fold , self.split_type) + file_type


        if self.data_path.endswith(".parquet"):
            self.data = self._load_parquet()
        elif self.data_path.endswith(".json"):
            self.data = self._load_json()
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path}，请提供 .parquet 或 .json 文件")

    def _load_parquet(self) -> dict:
        """
        读取 Parquet 文件并返回 Dict。
        """
        import pandas as pd
        try:
            df = pd.read_parquet(self.data_path)
            return df.to_dict(orient="records") 
        except Exception as e:
            print(f"读取 Parquet 文件时出错: {e}")
            return []

    def _load_json(self) -> dict:
        """
        读取 JSON 文件并返回 Dict。
        """
        try:
            import json
            with open(self.data_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"读取 JSON 文件时出错: {e}")
            return []

    def _build_prompt(self, item: Dict[str, Any], template: Dict[str, str]) -> str:
        """
        构建提示词
        
        Args:
            item: 数据项
            template: 提示词模板
            
        Returns:
            构建好的提示词 prompts, images(list[image1, image2, ...])
        """
        return self._replace_images_tokens(template["user"].format(**item))

    def _load_yaml(self,prompt_path):
        """
            下载不同task的prompt
        """
        import yaml
        with open(prompt_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data["prompts"][self.dataname]
        
    def _parse_options(self, options:List[str]):
        """
        将options加上选项ABCD
        """

        option_letters = [chr(ord("A") + i) for i in range(len(options))]
        choices_str = "\n".join([f"{option_letter}. {option}" for option_letter, option in zip(option_letters, options)])
        return choices_str
    
    def _replace_images_tokens(self , input_string):
        image_order = [int(num) for num in re.findall(r"<image\s+(\d+)>", input_string)]
        input_string = re.sub(r"<image\s+\d+>", "[image]", input_string)

        return input_string ,image_order

    def build_input(self , batch: List[Dict], prompt_path):
        """
        构建一批输入，处理多图片情况
        这里的batch是指模型下载
        """
        self.prompt_template = self._load_yaml(prompt_path)
        inputs = []
        for item in batch:
            item["options"] = self._parse_options(ast.literal_eval(str(item["options"])))


            prompts , image_order = self._build_prompt(item, self.prompt_template)
            images = []
            # 根据MMMU数据集，最多有7张图片
            for i in image_order:
                image_key = f"image_{i}_path"
                if image_key in item:
                    img_path = os.path.join(self.data_fold, item[image_key])
                    img = Image.open(img_path).convert('RGB')
                    images.append(img)

            if len(images) == 1:
                multi_modal_data = {"image": images[0]}
            else:
                multi_modal_data = {"images": images}

            inputs.append({
                "prompt": prompts,
                "multi_modal_data": multi_modal_data
            })
        return inputs

    def save_indill(self, response):
        """
        30个类别得到结果后,将response传入来分开不同的类
        """
        
        # if os.path.exists(self.json_path):
        #     try:
        #         with open(self.json_path, 'r', encoding='utf-8') as f:
        #             config = json.load(f)
        #     except Exception as e:
        #         print(f"加载配置文件失败: {e}")
        pass


if __name__ == "__main__":
    mmmu = MMMUDataset("data/MMMU" , "dev")
    print(mmmu.build_input(mmmu.data[0:10] , "src/config/prompts_config.yaml"))

