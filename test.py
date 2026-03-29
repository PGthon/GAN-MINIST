from main import generate_and_save_images, load_and_generate
from model import  Generator

# 随机噪声的维度，你可以理解为"种子"
# 生成器接收这个100维的随机向量，输出28x28的图片
latent_dim = 100

# 训练轮数 - 完整遍历数据集的次数
# MNIST比较简单，50轮就能生成不错的数字
epochs = 20

# 图片大小 (MNIST是28x28)
img_shape = 28 * 28  # 784
load_and_generate()
