from vllm import LLM, SamplingParams
from PIL import Image
from typing import List, Optional, Union

from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor

# Qwen2.5-VL
def run_qwen2_5_vl(model_name, modality: str, gpu_num: int, max_image_per_prompt: int=1):
    
    llm = LLM(
        model=model_name,
        max_model_len=4096,
        max_num_seqs=5,
        mm_processor_kwargs={
            "min_pixels": 28 * 28,
            "max_pixels": 1280 * 28 * 28,
            "fps": 1,
        },
        disable_mm_preprocessor_cache=True,
        tensor_parallel_size=gpu_num,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        dtype="bfloat16",
        # enable_lora=True,
        limit_mm_per_prompt={"image": max_image_per_prompt}
    )

    if modality == "image":
        placeholder = "<|image_pad|>"
    elif modality == "video":
        placeholder = "<|video_pad|>"

    prompt_template = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
              f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
              "{query}<|im_end|>\n"
              "<|im_start|>assistant\n")
    stop_token_ids = None
    return llm, prompt_template, stop_token_ids


class QwenVLRunner:
    def __init__(self, model_name, gpu_num: int, modality: str='image', max_image_per_prompt: int=1):

        self.llm, self.prompt_template, self.stop_token_ids = run_qwen2_5_vl(model_name, modality, gpu_num, max_image_per_prompt=max_image_per_prompt)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.sampling_params = SamplingParams(temperature=0.85, max_tokens=2048, stop_token_ids=self.stop_token_ids) # length as the training settings

    # def generate(self, queries: List[str], images_path: List[str] = None, video_path: List[str] = None):

    #     inputs = [
    #             {"prompt": self.prompt_template.format(query=query), 
    #              "multi_modal_data": {
    #                 "image": Image.open(image_path).convert('RGB')
    #              }
    #             }
    #              for query, image_path in zip(queries, images_path)
    #     ]
        
    #     outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)

    #     batch_results = [item.outputs[0].text for item in outputs]
        
    #     return batch_results
    
    def generate(self, queries: List[str], images_path: List[List[Union[str, Image.Image]]] = None, video_path: Optional[List[str]] = None):
        """
            Genetrate results for multiple images
       """ 
        image_inputs_list = [] 
        prompt_list = []
        for query, image_paths in zip(queries, images_path):
            if not isinstance(image_paths, list):
                image_paths = [image_paths]
            placeholders = [{"type": "image", "image": image_path} for image_path in image_paths]
            messages = [
                # {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    *placeholders, 
                    {"type": "text", "text": query}
                ]},
            ]
            prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_list.append(prompt)
            image_data, _ = process_vision_info(messages)
            image_inputs_list.append(image_data)

        inputs = [
            {"prompt": prompt, "multi_modal_data": {"image": image_data}}
            for prompt, image_data in zip(prompt_list, image_inputs_list)
        ]
        outputs = self.llm.generate(inputs, sampling_params=self.sampling_params)
        batch_results = [item.outputs[0].text for item in outputs]
        return batch_results

if __name__ == "__main__":

    runner = QwenVLRunner(model_name="/home/nfs05/liyt/models/Qwen2.5-VL-7B-Instruct", gpu_num=1, modality="image", max_image_per_prompt=4)
    results = runner.generate(["What is the difference these images?"], 
        [["/home/nfs06/liyt/easy-vlmeval/data/MathVista/images/5845.jpg",
         "/home/nfs06/liyt/easy-vlmeval/data/MathVista/images/5842.jpg"]])
    print(results)
    
    results = runner.generate(["What is in the image?"], 
        ["/home/nfs06/liyt/easy-vlmeval/data/MathVista/images/5842.jpg"])
    print(results)