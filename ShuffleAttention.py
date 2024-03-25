import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter


class ShuffleAttention(nn.Module):
    # 初始化Shuffle Attention模块
    def __init__(self, channel=512, reduction=16, G=8):
        super().__init__()
        self.G = G  # 分组数量
        self.channel = channel  # 通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，用于生成通道注意力
        self.gn = nn.GroupNorm(channel // (2 * G), channel // (2 * G))  # 分组归一化，用于空间注意力
        # 以下为通道注意力和空间注意力的权重和偏置参数
        self.cweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.cbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sweight = Parameter(torch.zeros(1, channel // (2 * G), 1, 1))
        self.sbias = Parameter(torch.ones(1, channel // (2 * G), 1, 1))
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数，用于生成注意力图

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    # 通道混洗方法，用于在分组处理后重组特征
    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(b, -1, h, w)
        return x

    # 前向传播方法
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b * self.G, -1, h, w)  # 将输入特征图按照分组维度进行重排

        x_0, x_1 = x.chunk(2, dim=1)  # 将特征图分为两部分，分别用于通道注意力和空间注意力

        # 通道注意力分支
        x_channel = self.avg_pool(x_0)  # 对第一部分应用全局平均池化
        x_channel = self.cweight * x_channel + self.cbias  # 应用学习到的权重和偏置
        x_channel = x_0 * self.sigmoid(x_channel)  # 通过sigmoid激活函数和原始特征图相乘，得到加权的特征图

        # 空间注意力分支
        x_spatial = self.gn(x_1)  # 对第二部分应用分组归一化
        x_spatial = self.sweight * x_spatial + self.sbias  # 应用学习到的权重和偏置
        x_spatial = x_1 * self.sigmoid(x_spatial)  # 通过sigmoid激活函数和原始特征图相乘，得到加权的特征图

        # 将通道注意力和空间注意力的结果沿通道维度拼接
        out = torch.cat([x_channel, x_spatial], dim=1)
        out = out.contiguous().view(b, -1, h, w)  # 重新调整形状以匹配原始输入的维度

        # 应用通道混洗，以便不同分组间的特征可以交换信息
        out = self.channel_shuffle(out, 2)
        return out


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    se = ShuffleAttention(channel=512, G=8)
    output = se(input)
    print(output.shape)