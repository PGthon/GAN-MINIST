# cGAN（条件生成对抗网络）训练文件
# 可以指定生成某个数字（0-9）

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
from dataloader import get_dataloader, labels_to_onehot
# 导入模型模块
from model import cGAN_Generator, cGAN_Discriminator

# 随机噪声的维度
latent_dim = 256

# 数字类别数量（0-9共10个）
num_classes = 10

# 批量大小
batch_size = 256

# 学习率
lr = 0.0002

# 训练轮数
epochs = 50

# 限制训练样本数量
limit_samples = 60000

# 图片大小 (MNIST是28x28)
img_shape = 28 * 28  # 784

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# 生成函数：生成指定数字的图片
def generate_digit(generator, digit, n_samples=8):
    """生成指定数字的图片

    Args:
        generator: 生成器模型
        digit: 要生成的数字 (0-9)
        n_samples: 生成几张图片
    """
    # 创建数字标签（one-hot编码）
    labels = torch.full((n_samples,), digit, dtype=torch.long)
    labels_onehot = labels_to_onehot(labels, num_classes).to(device)

    # 生成随机噪声
    z = torch.randn(n_samples, latent_dim).to(device)

    # 生成图片
    with torch.no_grad():
        generated_imgs = generator(z, labels_onehot)

    # 转换为numpy数组并反归一化
    generated_imgs = generated_imgs.cpu().numpy()
    generated_imgs = 0.5 * generated_imgs + 0.5

    # 显示图片
    fig, axes = plt.subplots(1, n_samples, figsize=(16, 2))
    for i, ax in enumerate(axes):
        ax.imshow(generated_imgs[i].squeeze(), cmap='gray')
        ax.axis('off')
        ax.set_title(f'生成数字 {digit}')
    plt.tight_layout()
    plt.savefig(f'generated_digit_{digit}.png')
    plt.show()


# 生成所有数字0-9的图片
def generate_all_digits(generator, n_samples=5):
    """生成所有数字0-9的图片"""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for digit in range(10):
        # 创建数字标签
        labels = torch.full((n_samples,), digit, dtype=torch.long)
        labels_onehot = labels_to_onehot(labels, num_classes).to(device)

        # 生成随机噪声
        z = torch.randn(n_samples, latent_dim).to(device)

        # 生成图片
        with torch.no_grad():
            generated_imgs = generator(z, labels_onehot)

        # 转换为numpy
        generated_imgs = generated_imgs.cpu().numpy()
        generated_imgs = 0.5 * generated_imgs + 0.5

        # 显示第一张
        axes[digit].imshow(generated_imgs[0].squeeze(), cmap='gray')
        axes[digit].axis('off')
        axes[digit].set_title(f'数字 {digit}')

    plt.tight_layout()
    plt.savefig('cgan_all_digits.png')
    plt.show()


# 训练函数
def train():
    # 记录损失值
    d_losses = []
    g_losses = []
    steps = []

    # 初始化cGAN模型
    generator = cGAN_Generator(latent_dim, num_classes, img_shape)
    discriminator = cGAN_Discriminator(num_classes, img_shape)

    # 将模型移到GPU上
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    # 定义损失函数
    criterion = nn.BCELoss()

    # 定义优化器
    optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

    # 获取数据加载器
    train_loader, _ = get_dataloader(batch_size, limit_samples)

    # 训练循环
    step_count = 0
    for epoch in range(epochs):
        for i, (imgs, labels) in enumerate(train_loader):
            batch_size_current = imgs.size(0)

            # 将数据和标签移到GPU上
            imgs = imgs.to(device)
            labels_onehot = labels_to_onehot(labels, num_classes).to(device)

            # 创建真实和假标签
            real_labels = torch.ones(batch_size_current, 1).to(device)
            fake_labels = torch.zeros(batch_size_current, 1).to(device)

            # ==================== 训练判别器 ====================
            optimizer_D.zero_grad()

            # 判别器对真实图片+对应标签的判断（应该接近1）
            real_outputs = discriminator(imgs, labels_onehot)
            d_loss_real = criterion(real_outputs, real_labels)

            # 生成随机噪声和随机标签
            z = torch.randn(batch_size_current, latent_dim).to(device)
            # 随机标签用于生成器生成图片
            random_labels = torch.randint(0, num_classes, (batch_size_current,))
            random_labels_onehot = labels_to_onehot(random_labels, num_classes).to(device)

            # 生成假图片
            fake_imgs = generator(z, random_labels_onehot)

            # 判别器对假图片+输入标签的判断（应该接近0）
            fake_outputs = discriminator(fake_imgs.detach(), random_labels_onehot)
            d_loss_fake = criterion(fake_outputs, fake_labels)

            # 判别器总损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # ==================== 训练生成器 ====================
            optimizer_G.zero_grad()

            # 重新生成随机噪声和标签
            z = torch.randn(batch_size_current, latent_dim).to(device)
            random_labels = torch.randint(0, num_classes, (batch_size_current,))
            random_labels_onehot = labels_to_onehot(random_labels, num_classes).to(device)

            # 生成假图片
            fake_imgs = generator(z, random_labels_onehot)

            # 判别器对假图片+对应标签的判断（生成器希望判别器输出接近1）
            fake_outputs = discriminator(fake_imgs, random_labels_onehot)
            g_loss = criterion(fake_outputs, real_labels)

            g_loss.backward()
            optimizer_G.step()

            # 记录损失值
            step_count += 1
            if step_count % 50 == 0:
                d_losses.append(d_loss.item())
                g_losses.append(g_loss.item())
                steps.append(step_count)
                print(f'Epoch [{epoch+1}/{epochs}], Step [{step_count}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(steps, d_losses, label='D Loss', alpha=0.7)
    plt.plot(steps, g_losses, label='G Loss', alpha=0.7)
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('cGAN Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('cgan_training_loss.png')
    plt.show()

    # 保存模型
    torch.save(generator.cpu().state_dict(), 'cgan_generator.pth')
    torch.save(discriminator.cpu().state_dict(), 'cgan_discriminator.pth')
    print('cGAN模型已保存！')

    # 移回GPU并生成图片
    generator = generator.to(device)

    # 生成所有数字0-9的图片
    generate_all_digits(generator)


# 加载模型并生成指定数字的图片
def load_and_generate():
    # 初始化模型
    generator = cGAN_Generator(latent_dim, num_classes, img_shape)

    # 加载权重
    generator.load_state_dict(torch.load('cgan_generator.pth'))
    generator = generator.to(device)
    generator.eval()
    print('cGAN模型加载成功！')

    # 生成指定数字（比如生成数字5）
    print('\n=== 生成数字 0 ===')
    generate_digit(generator, 0)

    print('\n=== 生成数字 1 ===')
    generate_digit(generator, 1)

    print('\n=== 生成数字 5 ===')
    generate_digit(generator, 5)

    # 生成所有数字
    print('\n=== 生成所有数字 0-9 ===')
    generate_all_digits(generator)


# 主程序入口
if __name__ == '__main__':
    # 训练模式
    train()

    # 加载模型生成图片模式（取消下面注释）
    # load_and_generate()
