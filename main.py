# 导入PyTorch核心库
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入优化器
import torch.optim as optim
# 导入可视化工具
import matplotlib.pyplot as plt
# 导入数值计算库
import numpy as np

# 导入数据加载器模块
from dataloader import get_dataloader
# 导入模型模块
from model import Generator, Discriminator

# 随机噪声的维度，你可以理解为"种子"
# 生成器接收这个100维的随机向量，输出28x28的图片
latent_dim = 100

# 批量大小 - 每次训练用多少张图片
batch_size = 512

# 学习率 - 控制模型学习的步长
lr = 0.0002

# 训练轮数 - 完整遍历数据集的次数
epochs = 10

# 图片大小 (MNIST是28x28)
img_shape = 28 * 28  # 784

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 生成函数：生成并显示图像
def generate_and_save_images(generator, latent_dim, n_samples=16):
    # 生成随机噪声
    z = torch.randn(n_samples, latent_dim).to(device)
    # 生成图像
    with torch.no_grad():
        generated_imgs = generator(z)

    # 转换为numpy数组并反归一化
    generated_imgs = generated_imgs.cpu().numpy()
    generated_imgs = 0.5 * generated_imgs + 0.5  # 将[-1, 1]转换为[0, 1]

    # 创建4x4的图像网格
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        # 显示图像
        ax.imshow(generated_imgs[i].squeeze(), cmap='gray')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('generated_images.png')
    plt.show()


# 训练函数
def train(batch_size=None):
    # 初始化生成器
    generator = Generator(latent_dim, img_shape)
    # 初始化判别器
    discriminator = Discriminator(img_shape)

    # 将模型移到GPU上
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 定义损失函数
    criterion = nn.BCELoss()

    # 定义优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 获取数据加载器
    train_loader, _ = get_dataloader(batch_size)

    # 训练循环
    for epoch in range(epochs):
        for i, (imgs, _) in enumerate(train_loader):
            batch_size = imgs.size(0)

            # 将数据移到GPU上
            imgs = imgs.to(device)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 训练判别器
            optimizer_D.zero_grad()
            real_outputs = discriminator(imgs)
            d_loss_real = criterion(real_outputs, real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_outputs = discriminator(fake_imgs.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            fake_outputs = discriminator(fake_imgs)
            g_loss = criterion(fake_outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # 保存模型
    torch.save(generator.cpu().state_dict(), 'generator.pth')
    torch.save(discriminator.cpu().state_dict(), 'discriminator.pth')
    print('模型已保存！')

    # 移回GPU并生成图片
    generator = generator.to(device)
    generate_and_save_images(generator, latent_dim)


# 加载模型并生成图片的函数
def load_and_generate():
    # 初始化模型
    generator = Generator(latent_dim, img_shape)

    # 加载权重
    generator.load_state_dict(torch.load('generator.pth'))
    generator = generator.to(device)
    generator.eval()
    print('模型加载成功！')

    # 生成图片
    generate_and_save_images(generator, latent_dim)


# 主程序入口
if __name__ == '__main__':
    # 运行模式选择：
    # 模式1：训练模型（默认）
    train()

    # 模式2：加载已有模型生成图片（取消下面注释）
    # load_and_generate()
