import os
import sys
import inspect

from torch import nn
import torch


class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div, forward):
        """
        初始化函数
        :param dim: 输入通道的维度
        :param n_div: 输入维度划分的份数，用于确定哪一部分通道会应用卷积
        :param forward: 指定前向传播的模式，'slicing' 或 'split_cat'
        """
        super().__init__()
        self.dim_conv3 = dim // n_div  # 应用卷积的通道数
        self.dim_untouched = dim - self.dim_conv3  # 保持不变的通道数
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)  # 部分应用的3x3卷积

        # 根据forward参数，选择前向传播的方式
        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        """
        利用slicing方法的前向传播，主要用于推理
        :param x: 输入特征图
        :return: 输出特征图，部分通道被卷积处理
        """
        x = x.clone()  # 克隆输入以保持原输入不变，用于后续的残差连接
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        """
        利用split和cat方法的前向传播，可用于训练/推理
        :param x: 输入特征图
        :return: 输出特征图，部分通道被卷积处理，剩余通道保持不变
        """
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)  # 将输入特征图分为两部分
        x1 = self.partial_conv3(x1)  # 对第一部分应用卷积
        x = torch.cat((x1, x2), 1)  # 将处理后的第一部分和未处理的第二部分拼接
        return x


if __name__ == '__main__':
    block = Partial_conv3(64, 2, 'split_cat').cuda()  # 实例化模型
    input = torch.rand(1, 64, 64, 64).cuda()  # 创建输入张量
    output = block(input)  # 执行前向传播
    print(output.shape)  # 输出的尺寸