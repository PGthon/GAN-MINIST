# 导入PyTorch核心库
import torch
# 导入神经网络模块
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        # 调用父类的初始化方法
        super(Generator, self).__init__()
        # 定义生成器的网络结构
        self.model = nn.Sequential(
            # 第一层：输入噪声向量，输出256维
            nn.Linear(latent_dim, 256),
            # ReLU激活函数，引入非线性
            nn.ReLU(),
            # 第二层：输出512维
            nn.Linear(256, 512),
            nn.ReLU(),
            # 第三层：输出1024维
            nn.Linear(512, 1024),
            nn.ReLU(),
            # 输出层：输出图像大小
            nn.Linear(1024, img_shape),
            # tanh激活函数，将输出限制在[-1, 1]，与数据归一化范围一致
            nn.Tanh()
        )

    def forward(self, z):
        # 前向传播：将噪声向量输入网络
        img = self.model(z)
        # 重塑为图像形状 (batch_size, 1, 28, 28)
        # 1表示通道数（灰度图），28x28是图像大小
        img = img.view(img.size(0), 1, 28, 28)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        # 调用父类的初始化方法
        super(Discriminator, self).__init__()
        # 定义判别器的网络结构
        self.model = nn.Sequential(
            # 输入层：将图像展开为一维向量
            nn.Linear(img_shape, 1024),
            # LeakyReLU激活函数，防止梯度消失，斜率为0.2
            nn.LeakyReLU(0.2),
            # 第二层
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            # 第三层
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            # 输出层：二分类概率
            nn.Linear(256, 1),
            # sigmoid激活函数，将输出映射到0-1之间的概率
            nn.Sigmoid()
        )

    def forward(self, img):
        # 将图像展开为一维向量
        img_flat = img.view(img.size(0), -1)
        # 前向传播：将展开的图像输入网络
        validity = self.model(img_flat)
        return validity


# ==================== cGAN 模型 ====================
# 条件GAN：输入噪声+标签，输出指定标签的图片

class cGAN_Generator(nn.Module):
    def __init__(self, latent_dim, num_classes, img_shape):
        super(cGAN_Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        # 噪声和标签的embedding层
        # 将噪声和标签拼接后的总输入维度 = latent_dim + num_classes
        input_dim = latent_dim + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_shape),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # 将噪声和标签拼接
        # z: (batch_size, latent_dim)
        # labels: (batch_size, num_classes) one-hot编码
        gen_input = torch.cat([z, labels], dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 28, 28)
        return img


class cGAN_Discriminator(nn.Module):
    def __init__(self, num_classes, img_shape):
        super(cGAN_Discriminator, self).__init__()
        self.num_classes = num_classes

        # 图片和标签拼接后的总输入维度 = img_shape + num_classes
        input_dim = img_shape + num_classes

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # 将图片和标签拼接
        # img: (batch_size, 1, 28, 28)
        # labels: (batch_size, num_classes) one-hot编码
        img_flat = img.view(img.size(0), -1)
        d_input = torch.cat([img_flat, labels], dim=1)
        validity = self.model(d_input)
        return validity