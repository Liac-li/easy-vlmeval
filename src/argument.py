# argument.py

import argparse
import yaml

def add_loadData_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group('loaddata', 'loaddata configurations')
    group.add_argument('--config', type = str ,  help='Path to the YAML config file')
    
    return parser

def get_args():
    parser = argparse.ArgumentParser()
    parser = add_loadData_args(parser)
    args, unknown = parser.parse_known_args()

    # 读取 YAML 文件
    with open(args.config, 'r') as f:
        yaml_config = yaml.safe_load(f)

    # 把第一层 key（如 MathVista）作为任务名
    task_name = list(yaml_config.keys())[0]
    task_config = yaml_config[task_name]

    return task_name, task_config
