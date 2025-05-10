# Easy-VLMEval

一个专注于视觉语言模型（VLM）评测的简单易用工具，支持多种主流基准测试，便于快速集成和扩展。

## 功能特点

- 支持多种视觉语言基准测试数据集（如 MMMU、MathVista 等），基于官方数据集格式，无需修改原始 prompt 逻辑
- 支持基于 vLLM 的本地 API 服务推理与评测
- 灵活的提示词模板与模型参数配置
- 简单高效的评测流程，自动化推理与结果评估

## 目录结构

```
easy-vlmeval/
├── data/                  # 数据集存放目录（需手动下载）
│   └── <dataset_name>/    # 如 MMMU、MathVista 等
├── outputs/               # 推理与评测结果输出目录
├── scripts/               # 常用运行脚本
├── src/                   # 核心源代码
│   ├── benchmarks/        # 各基准测试相关代码
│   ├── config/            # 配置管理
│   ├── evaluator/         # 评测流程与评测器
│   ├── examples/          # 示例代码（如 loaddata/）
│   ├── tasks/             # 任务定义
│   ├── utils/             # 工具函数
│   └── run_vllm.py        # vLLM 推理主入口
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明文档
```

## 安装方法

```bash
pip install -r requirements.txt
```

## 数据集准备

1. 在 `data/` 目录下新建对应数据集文件夹，并下载官方数据集（如 MMMU、MathVista 等）。
2. 解压所需图片压缩包，保持原始目录结构，无需修改。
3. 以 MMMU 为例，下载命令如下：

```bash
cd data
huggingface-cli download --resume-download MMMU/MMMU --local-dir MMMU
```

4. 在 `src/config/datasets_config.yaml` 中配置数据集本地路径（以 `data/` 目录为基准）。

## 运行与评测

以 vLLM 本地推理为例，假设当前路径为项目根目录：

```bash
python src/run_vllm.py --modal-path PATH_TO_MODEL_DIR --gpu-num 1 --dataset mmmu@validation --prompt r1_think_prompt --output_dir outputs --eval
```

- `--modal-path`：模型目录路径
- `--gpu-num`：使用 GPU 数量
- `--dataset`：数据集及子集（如 mmmu@validation）
- `--prompt`：提示词模板名称
- `--output_dir`：输出目录
- `--eval`：自动评测推理结果

## 配置说明

- 在 `configs/prompts/` 目录下创建和管理提示词模板。
- 在 `configs/models/` 目录下配置模型参数。

## 数据格式说明

推理与评测过程中，数据样例如下：

```python
converted_data = {
    "id": sample['pid'],
    "query": sample['query'],
    "answer": sample['answer'],
    "choices": sample['choices'] if sample['choices'] else None,
    "image": {
        "path": "absolute_path_to_image",
        "bytes": "bytes_encoded_image",
        "item": PIL.Image
    },
    "origin_data": sample  # 保留原始数据
}
```