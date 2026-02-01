import torch

# 1. 检查CUDA是否可用
print(f"CUDA是否可用: {torch.cuda.is_available()}")
# 2. 检查可用GPU数量
print(f"可用GPU数量: {torch.cuda.device_count()}")
# 3. 查看PyTorch版本
print(f"PyTorch版本: {torch.__version__}")
# 4. 查看默认张量设备
print(f"默认张量设备: {torch.tensor([1,2,3]).device}")

# 可选：如果有GPU，打印详细信息
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
else:
    print("当前环境使用的是CPU版的PyTorch。")