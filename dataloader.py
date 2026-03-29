# 导入PyTorch核心库
import torch
# 导入torchvision库，包含常用数据集和模型
import torchvision
# 导入数据变换模块
import torchvision.transforms as transforms
# 导入数据加载器
from torch.utils.data import DataLoader

# 定义数据变换
# Compose用于组合多个变换操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 将PIL图像转换为PyTorch张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化到 [-1, 1] 范围
])

# 加载MNIST训练数据集
train_dataset = torchvision.datasets.MNIST(
    root='./data',  # 数据集存储路径
    train=True,     # 加载训练集
    download=True,  # 自动下载
    transform=transform  # 应用数据变换
)

# 加载MNIST测试数据集
test_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=False,  # 加载测试集
    download=True,
    transform=transform
)

# 创建数据加载器函数
def get_dataloader(batch_size):
    # 训练数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,  # 批量大小
        shuffle=True,  # 打乱数据
        num_workers=0  # 多线程加载
    )
    # 测试数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False  # 测试集不需要打乱
    )
    return train_loader, test_loader