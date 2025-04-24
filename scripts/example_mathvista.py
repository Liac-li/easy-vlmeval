"""
MathVista数据集使用示例
"""

from src.datasets.mathvista import MathVistaDataset

def main():
    # 从本地加载testmini数据集
    dataset = MathVistaDataset(split="testmini")
    print(f"数据集大小: {len(dataset)}")
    
    # 获取第一个样本并转换为推理格式
    converted_sample = dataset.convert_to_inference_format(0)
    print("\n转换后的样本:")
    print(f"ID: {converted_sample['id']}")
    print(f"Query: {converted_sample['query'][:100]}...")  # 只打印前100个字符
    print(f"Answer: {converted_sample['answer']}")
    print(f"Choices: {converted_sample['choices']}")
    print(f"Image path: {converted_sample['image']['path']}")
    print(f"Image size: {len(converted_sample['image']['bytes'])} bytes")
    
    # 批量转换示例
    print("\n批量转换前5个样本:")
    for i in range(5):
        converted = dataset.convert_to_inference_format(i)
        print(f"样本 {i}: ID={converted['id']}, Answer={converted['answer']}")

    # 获取一批样本
    batch = dataset.get_batch(0, 5)
    print(f"\n前5个样本的问题ID: {[item['pid'] for item in batch]}")
    
    # 尝试加载test数据集
    try:
        test_dataset = MathVistaDataset(split="test")
        print(f"\ntest数据集大小: {len(test_dataset)}")
    except FileNotFoundError:
        print("\n未找到test数据集，请确保已下载")

if __name__ == "__main__":
    main() 