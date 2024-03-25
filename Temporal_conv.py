import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


# 膨胀卷积模块，包含了膨胀卷积和一个前馈网络
class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor, seq_len):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()  # 用于存储多个膨胀卷积层
        self.padding = 0  # 卷积层的填充值
        self.seq_len = seq_len  # 输入序列的长度
        self.kernel_set = [2, 3, 6, 7]  # 定义一组不同大小的卷积核
        cout = int(cout / len(self.kernel_set))  # 计算每个卷积层的输出通道数
        for kern in self.kernel_set:  # 为每个卷积核大小创建一个膨胀卷积层
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

        # 通过线性层调整时间维度的大小
        self.out = nn.Sequential(
            nn.Linear(self.seq_len - dilation_factor * (self.kernel_set[-1] - 1) + self.padding * 2 - 1 + 1, cin),
            nn.ReLU(),
            nn.Linear(cin, self.seq_len)
        )

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))  # 对输入应用每个膨胀卷积层
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]  # 根据最短序列调整序列长度，以确保所有输出具有相同的时间维度长度

        x = torch.cat(x, dim=1)  # 将所有膨胀卷积层的输出沿通道维度拼接
        x = self.out(x)  # 通过前馈网络调整时间序列的长度
        return x


# 时间卷积网络，整合了滤波器和门控机制
class temporal_conv(nn.Module):
    def __init__(self, cin, cout, dilation_factor, seq_len):
        super(temporal_conv, self).__init__()

        self.filter_convs = dilated_inception(cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len)
        self.gated_convs = dilated_inception(cin=cin, cout=cout, dilation_factor=dilation_factor, seq_len=seq_len)

    def forward(self, X):
        filter = self.filter_convs(X)  # 计算滤波器分支的输出
        filter = torch.tanh(filter)  # 对滤波器分支应用tanh激活函数
        gate = self.gated_convs(X)  # 计算门控分支的输出
        gate = torch.sigmoid(gate)  # 对门控分支应用sigmoid函数
        out = filter * gate  # 将滤波器分支和门控分支的输出相乘，实现门控机制
        return out


if __name__ == '__main__':
    X = torch.randn(1, 32, 1, 24)  # 示例输入
    Model = temporal_conv(cin=32, cout=32, dilation_factor=1, seq_len=24)  # 实例化模型
    out = Model(X)  # 计算输出
    print(out.shape)  # 打印输出形状