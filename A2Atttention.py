# A2-Nets: Double Attention Networks
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F


class DoubleAttention(nn.Module):

    def __init__(self, in_channels, c_m, c_n, reconstruct=True):
        super().__init__()
        self.in_channels = in_channels  # 输入通道数
        self.reconstruct = reconstruct  # 是否需要重构输出以匹配输入的维度
        self.c_m = c_m  # 第一个注意力机制的输出通道数
        self.c_n = c_n  # 第二个注意力机制的输出通道数
        # 定义三个1x1卷积层，用于生成A、B和V特征
        self.convA = nn.Conv2d(in_channels, c_m, 1)
        self.convB = nn.Conv2d(in_channels, c_n, 1)
        self.convV = nn.Conv2d(in_channels, c_n, 1)
        # 如果需要重构，定义一个1x1卷积层用于输出重构
        if self.reconstruct:
            self.conv_reconstruct = nn.Conv2d(c_m, in_channels, kernel_size=1)
        self.init_weights()

    def init_weights(self):
        # 权重初始化
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

    def forward(self, x):
        # 前向传播
        b, c, h, w = x.shape
        assert c == self.in_channels  # 确保输入通道数与初始化时一致
        A = self.convA(x)  # b,c_m,h,w# 生成A特征图
        B = self.convB(x)  # b,c_n,h,w# 生成B特征图
        V = self.convV(x)  # b,c_n,h,w# 生成V特征图
        # 将特征图维度调整为方便矩阵乘法的形状
        tmpA = A.view(b, self.c_m, -1)
        attention_maps = F.softmax(B.view(b, self.c_n, -1))
        attention_vectors = F.softmax(V.view(b, self.c_n, -1))
        # 步骤1: 特征门控
        global_descriptors = torch.bmm(tmpA, attention_maps.permute(0, 2, 1))  # b.c_m,c_n
        # 步骤2: 特征分配
        tmpZ = global_descriptors.matmul(attention_vectors)  # b,c_m,h*w
        tmpZ = tmpZ.view(b, self.c_m, h, w)  # b,c_m,h,w
        if self.reconstruct:
            tmpZ = self.conv_reconstruct(tmpZ)  # 如果需要，通过重构层调整输出通道数

        return tmpZ


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = DoubleAttention(64, 128, 128)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(input.size(), output.size())