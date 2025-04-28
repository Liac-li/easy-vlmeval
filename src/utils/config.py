"""
配置工具模块
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = 'src/config') -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    config_names = ['datasets_config.yaml', 'models_config.yaml', 'prompts_config.yaml']
    config_paths = [config_path / name for name in config_names]
    config_dict = {name.split('_')[0]: {"path": path} for name, path in zip(config_names, config_paths)}

    for config_type in config_dict:
        config_path = config_dict[config_type]["path"]
        if not config_path.exists():
            raise ValueError(f"配置文件不存在: {config_path}")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            config_dict[config_type]["config"] = config

    return config_dict

if __name__ == "__main__":
    config = load_config()
    print(config['datasets']['config'])