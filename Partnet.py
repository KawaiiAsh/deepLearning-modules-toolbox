import numpy as np
import torch
from torch import nn
from torch.nn import init


class ParNetAttention(nn.Module):
    # 初始化ParNet注意力模块
    def __init__(self, channel=512):
        super().__init__()
        # 使用自适应平均池化和1x1卷积实现空间压缩，然后通过Sigmoid激活函数产生权重图
        self.sse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，将空间维度压缩到1x1
            nn.Conv2d(channel, channel, kernel_size=1),  # 1x1卷积，用于调整通道的权重
            nn.Sigmoid()  # Sigmoid函数，用于生成注意力图
        )

        # 通过1x1卷积实现特征重映射，不改变空间尺寸
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=1),  # 1x1卷积，不改变特征图的空间尺寸
            nn.BatchNorm2d(channel)  # 批量归一化
        )

        # 通过3x3卷积捕获空间上下文信息
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, padding=1),  # 3x3卷积，保持特征图尺寸不变
            nn.BatchNorm2d(channel)  # 批量归一化
        )

        self.silu = nn.SiLU()  # SiLU激活函数，也被称为Swish函数

    def forward(self, x):
        # x是输入的特征图，形状为(Batch, Channel, Height, Width)
        b, c, _, _ = x.size()
        x1 = self.conv1x1(x)  # 通过1x1卷积处理x
        x2 = self.conv3x3(x)  # 通过3x3卷积处理x
        x3 = self.sse(x) * x  # 应用空间压缩的注意力权重到x上
        y = self.silu(x1 + x2 + x3)  # 将上述三个结果相加并通过SiLU激活函数激活，获得最终输出
        return y


# 测试ParNetAttention模块
if __name__ == '__main__':
    input = torch.randn(3, 512, 7, 7)  # 创建一个随机输入
    pna = ParNetAttention(channel=512)  # 实例化ParNet注意力模块
    output = pna(input)  # 对输入进行处理
    print(output.shape)  # 打印输出的形状，预期为(3, 512, 7, 7)