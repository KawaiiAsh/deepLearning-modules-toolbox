import numpy as np
import torch
from torch import nn
from torch.nn import init
from collections import OrderedDict


class SKAttention(nn.Module):
    def __init__(self, channel=512, kernels=[1, 3, 5, 7], reduction=16, group=1, L=32):
        super().__init__()
        # 计算维度压缩后的向量长度
        self.d = max(L, channel // reduction)
        # 不同尺寸的卷积核组成的卷积层列表
        self.convs = nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv', nn.Conv2d(channel, channel, kernel_size=k, padding=k // 2, groups=group)),
                    ('bn', nn.BatchNorm2d(channel)),
                    ('relu', nn.ReLU())
                ]))
            )
        # 通道数压缩的全连接层
        self.fc = nn.Linear(channel, self.d)
        # 为每个卷积核尺寸对应的特征图计算注意力权重的全连接层列表
        self.fcs = nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d, channel))
        # 注意力权重的Softmax层
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs = []
        # 通过不同尺寸的卷积核处理输入
        for conv in self.convs:
            conv_outs.append(conv(x))
        feats = torch.stack(conv_outs, 0)  # k,bs,channel,h,w

        # 将所有卷积核的输出求和得到融合特征图U
        U = sum(conv_outs)  # bs,c,h,w

        # 对融合特征图U进行全局平均池化，并通过全连接层降维得到Z
        S = U.mean(-1).mean(-1)  # bs,c
        Z = self.fc(S)  # bs,d

        # 计算每个卷积核对应的注意力权重
        weights = []
        for fc in self.fcs:
            weight = fc(Z)
            weights.append(weight.view(bs, c, 1, 1))  # bs,channel
        attention_weights = torch.stack(weights, 0)  # k,bs,channel,1,1
        attention_weights = self.softmax(attention_weights)  # k,bs,channel,1,1

        # 将注意力权重应用到对应的特征图上，并对所有特征图进行加权求和得到最终的输出V
        V = (attention_weights * feats).sum(0)
        return V


# 示例用法
if __name__ == '__main__':
    input = torch.randn(50, 512, 7, 7)
    sk = SKAttention(channel=512, reduction=8)
    output = sk(input)
    print(output.shape)  # 输出经过SK注意力处理后的特征图形状