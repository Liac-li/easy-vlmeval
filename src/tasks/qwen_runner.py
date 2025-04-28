from vllm import LLM, SamplingParams
from PIL import Image
from typing import List

# Qwen2.5-VL
def run_qwen2_5_vl(model_name, modality: str, gpu_num: int):
    
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
        # enable_lora=True,
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
    def __init__(self, model_name, gpu_num: int, modality: str='image'):

        self.llm, self.prompt_template, self.stop_token_ids = run_qwen2_5_vl(model_name, modality, gpu_num)

    def generate(self, queries: List[str], images_path: List[str] = None, video_path: List[str] = None):

        sampling_params = SamplingParams(temperature=0.85, max_tokens=2048, stop_token_ids=self.stop_token_ids) # length as the training settings
        inputs = [
                {"prompt": self.prompt_template.format(query=query), 
                 "multi_modal_data": {
                    "image": Image.open(image_path).convert('RGB')
                 }
                }
                 for query, image_path in zip(queries, images_path)
        ]
        
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)

        batch_results = [item.outputs[0].text for item in outputs]
        
        return batch_results
