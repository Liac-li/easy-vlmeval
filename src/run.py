import argparse

from utils.config import load_config
from utils.logger import setup_logger


logger = setup_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="src/config", help="配置文件路径")
    parser.add_argument("--model", type=str, default="vllm", help="模型名称")
    parser.add_argument("--dataset", type=str, default="", help="数据集名称") # name@split, all lower
    parser.add_argument("--prompt", type=str, default="", help="提示词")
    parser.add_argument("--output-dir", type=str, default="", help="输出路径")
    parser.add_argument("--num-workers", type=int, default=4, help='进程数')
    
    parser.add_argument("--eval", action="store_true", help="是否评估")
    parser.add_argument("--reuse-saved", action="store_true", help="是否重用保存的结果")

    return parser.parse_args()

def main(args, config):
    logger.info(f"开始运行，配置文件路径: {args.config}")

    # load dataset
    
    # test model with dummy query
    
    # run inference
    
    # save results
    
    # evaluate results


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    main(args, config)



