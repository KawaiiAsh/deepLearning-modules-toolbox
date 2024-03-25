import numpy as np
import torch
from torch import nn
from torch.nn import init


# 定义外部注意力类，继承自nn.Module
class ExternalAttention(nn.Module):

    def __init__(self, d_model, S=64):
        super().__init__()
        # 初始化两个线性变换层，用于生成注意力映射
        # mk: 将输入特征从d_model维映射到S维，即降维到共享内存空间的大小
        self.mk = nn.Linear(d_model, S, bias=False)
        # mv: 将降维后的特征从S维映射回原始的d_model维
        self.mv = nn.Linear(S, d_model, bias=False)
        # 使用Softmax函数进行归一化处理
        self.softmax = nn.Softmax(dim=1)
        # 调用权重初始化函数
        self.init_weights()

    def init_weights(self):
        # 自定义权重初始化方法
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 对卷积层的权重进行Kaiming正态分布初始化
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    # 如果有偏置项，则将其初始化为0
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # 对批归一化层的权重和偏置进行常数初始化
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对线性层的权重进行正态分布初始化，偏置项（如果存在）初始化为0
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        # 前向传播函数
        attn = self.mk(queries)  # 使用mk层将输入特征降维到S维
        attn = self.softmax(attn)  # 对降维后的特征进行Softmax归一化处理
        # 对归一化后的注意力分数进行标准化，使其和为1
        attn = attn / torch.sum(attn, dim=2, keepdim=True)
        out = self.mv(attn)  # 使用mv层将注意力特征映射回原始维度
        return out


# 示例代码，创建一个ExternalAttention实例，并对一个随机输入进行处理
if __name__ == '__main__':
    block = ExternalAttention(d_model=64, S=8).cuda()  # 实例化模型并移至CUDA设备
    input = torch.rand(64, 64, 64).cuda()  # 创建随机输入
    output = block(input)  # 通过模型传递输入
    print(output.shape)  # 打印输入和输出的尺寸