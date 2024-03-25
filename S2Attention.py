import numpy as np
import torch
from torch import nn
from torch.nn import init


def spatial_shift1(x):
    # 实现第一种空间位移，位移图像的四分之一块
    b, w, h, c = x.size()
    # 以下四行代码分别向左、向右、向上、向下移动图像的四分之一块
    x[:, 1:, :, :c // 4] = x[:, :w - 1, :, :c // 4]
    x[:, :w - 1, :, c // 4:c // 2] = x[:, 1:, :, c // 4:c // 2]
    x[:, :, 1:, c // 2:c * 3 // 4] = x[:, :, :h - 1, c // 2:c * 3 // 4]
    x[:, :, :h - 1, 3 * c // 4:] = x[:, :, 1:, 3 * c // 4:]
    return x


def spatial_shift2(x):
    # 实现第二种空间位移，逻辑与spatial_shift1相似，但位移方向不同
    b, w, h, c = x.size()
    # 对图像的四分之一块进行空间位移
    x[:, :, 1:, :c // 4] = x[:, :, :h - 1, :c // 4]
    x[:, :, :h - 1, c // 4:c // 2] = x[:, :, 1:, c // 4:c // 2]
    x[:, 1:, :, c // 2:c * 3 // 4] = x[:, :w - 1, :, c // 2:c * 3 // 4]
    x[:, :w - 1, :, 3 * c // 4:] = x[:, 1:, :, 3 * c // 4:]
    return x


class SplitAttention(nn.Module):
    # 定义分割注意力模块，使用MLP层进行特征转换和注意力权重计算
    def __init__(self, channel=512, k=3):
        super().__init__()
        self.channel = channel
        self.k = k  # 分割的块数
        # 定义MLP层和激活函数
        self.mlp1 = nn.Linear(channel, channel, bias=False)
        self.gelu = nn.GELU()
        self.mlp2 = nn.Linear(channel, channel * k, bias=False)
        self.softmax = nn.Softmax(1)

    def forward(self, x_all):
        # 计算分割注意力，并应用于输入特征
        b, k, h, w, c = x_all.shape
        x_all = x_all.reshape(b, k, -1, c)  # 重塑维度
        a = torch.sum(torch.sum(x_all, 1), 1)  # 聚合特征
        hat_a = self.mlp2(self.gelu(self.mlp1(a)))  # 通过MLP计算注意力权重
        hat_a = hat_a.reshape(b, self.k, c)  # 调整形状
        bar_a = self.softmax(hat_a)  # 应用softmax获取注意力分布
        attention = bar_a.unsqueeze(-2)  # 增加维度
        out = attention * x_all  # 将注意力权重应用于特征
        out = torch.sum(out, 1).reshape(b, h, w, c)  # 聚合并调整形状
        return out


class S2Attention(nn.Module):
    # S2注意力模块，整合空间位移和分割注意力
    def __init__(self, channels=512):
        super().__init__()
        # 定义MLP层
        self.mlp1 = nn.Linear(channels, channels * 3)
        self.mlp2 = nn.Linear(channels, channels)
        self.split_attention = SplitAttention()

    def forward(self, x):
        b, c, w, h = x.size()
        x = x.permute(0, 2, 3, 1)  # 调整维度顺序
        x = self.mlp1(x)  # 通过MLP层扩展特征
        x1 = spatial_shift1(x[:, :, :, :c])  # 应用第一种空间位移
        x2 = spatial_shift2(x[:, :, :, c:c * 2])  # 应用第二种空间位移
        x3 = x[:, :, :, c * 2:]  # 保留原始特征的一部分
        x_all = torch.stack([x1, x2, x3], 1)  # 堆叠特征
        a = self.split_attention(x_all)  # 应用分割注意力
        x = self.mlp2(a)  # 通过另一个MLP层缩减特征维度
        x = x.permute(0, 3, 1, 2)  # 调整维度顺序回原始
        return x


# 示例代码
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)  # 创建输入张量
    s2att = S2Attention(channels=512)  # 实例化S2注意力模块
    output = s2att(input)  # 通过S2注意力模块处理输入
    print(output.shape)  # 打印输出张量的形状