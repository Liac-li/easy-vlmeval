import os
import yaml
from vllm import LLM, SamplingParams

def load_config(config_path="configs/models/default.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def start_vllm_service():
    config = load_config()
    
    # 初始化模型
    llm = LLM(
        model=config["model"]["name"],
        dtype=config["model"]["dtype"],
        max_model_len=config["model"]["max_model_len"],
        gpu_memory_utilization=config["model"]["gpu_memory_utilization"],
        tensor_parallel_size=config["vllm"]["tensor_parallel_size"],
        trust_remote_code=config["vllm"]["trust_remote_code"]
    )
    
    # 启动服务
    from vllm.entrypoints.api_server import run_server
    run_server(
        llm,
        host=config["vllm"]["host"],
        port=config["vllm"]["port"]
    )

if __name__ == "__main__":
    start_vllm_service() 