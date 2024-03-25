import torch
import torch.nn.functional as F
import torch.nn as nn


# GroupBatchnorm2d模块是对标准批量归一化的扩展，它将特征通道分组进行归一化。
class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num  # 确保通道数大于等于分组数
        self.group_num = group_num  # 分组数
        self.weight = nn.Parameter(torch.randn(c_num, 1, 1))  # 权重参数
        self.bias = nn.Parameter(torch.zeros(c_num, 1, 1))  # 偏置参数
        self.eps = eps  # 防止除以零

    def forward(self, x):
        # 实现分组批量归一化的前向传播
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)  # 根据分组数重新排列x的形状
        mean = x.mean(dim=2, keepdim=True)  # 计算每组的均值
        std = x.std(dim=2, keepdim=True)  # 计算每组的标准差
        x = (x - mean) / (std + self.eps)  # 归一化
        x = x.view(N, C, H, W)  # 恢复x的原始形状
        return x * self.weight + self.bias  # 应用权重和偏置


# SRU模块用于抑制空间冗余。它通过分组归一化和一个门控机制实现。
class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 torch_gn: bool = False
                 ):
        super().__init__()
        # 选择使用torch自带的GroupNorm还是自定义的GroupBatchnorm2d
        self.gn = nn.GroupNorm(num_channels=oup_channels, num_groups=group_num) if torch_gn else GroupBatchnorm2d(
            c_num=oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold  # 设置门控阈值
        self.sigomid = nn.Sigmoid()  # 使用Sigmoid函数作为激活函数

    def forward(self, x):
        # 实现SRU的前向传播
        gn_x = self.gn(x)  # 应用分组归一化
        w_gamma = self.gn.weight / torch.sum(self.gn.weight)  # 根据归一化权重计算重要性权重
        w_gamma = w_gamma.view(1, -1, 1, 1)
        reweigts = self.sigomid(gn_x * w_gamma)  # 计算重构权重
        # 根据门控阈值，将特征图分为信息丰富和信息较少的两部分
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * gn_x
        x_2 = noninfo_mask * gn_x
        x = self.reconstruct(x_1, x_2)  # 重构特征图
        return x

    def reconstruct(self, x_1, x_2):
        # 实现特征图的重构
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)  # 将信息丰富的特征图分为两部分
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)  # 将信息较少的特征图分为两部分
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)  # 通过特定方式合并特征图，增强特征表达


# CRU模块用于处理通道冗余。它通过一个压缩-卷积-扩展策略来增强特征的代表性。
class CRU(nn.Module):
    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)  # 计算上分支的通道数
        self.low_channel = low_channel = op_channel - up_channel  # 计算下分支的通道数
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)  # 上分支的压缩层
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)  # 下分支的压缩层
        # 上分支的卷积层，包括分组卷积和点卷积
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # 下分支的卷积层
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层

    def forward(self, x):
        # 实现CRU的前向传播
        # 将输入特征图分为上下两部分
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)  # 对上下两部分分别应用压缩层
        # 对上分支应用卷积层
        Y1 = self.GWC(up) + self.PWC1(up)
        # 对下分支应用卷积层，并与压缩后的低分支特征图合并
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # 合并上下分支的特征图，并应用自适应平均池化和softmax函数
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)  # 将合并后的特征图分为两部分
        return out1 + out2  # 将两部分的特征图相加，得到最终的输出


# ScConv模块结合了SRU和CRU两个子模块，用于同时处理空间和通道冗余。
class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 4,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,  # 初始化空间重构单元
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,  # 初始化通道重构单ู
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)  # 通过SRU处理空间冗余
        x = self.CRU(x)  # 通过CRU处理通道冗余
        return x  # 返回处理后的特征图


# 测试ScConv模块
if __name__ == '__main__':
    x = torch.randn(1, 32, 16, 16)  # 创建一个随机的输入张量
    model = ScConv(32)  # 创建一个ScConv模块实例
    print(model(x).shape)  # 打印ScConv处理后的输出张量形状