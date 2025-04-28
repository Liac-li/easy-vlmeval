#!/bin/bash

# 使用方法：bash run_download.sh configs/mathvista.yaml

CONFIG_PATH=$1

echo "🚀 Using config: $CONFIG_PATH"
python run_download.py --config $CONFIG_PATH
