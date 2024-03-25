import torch
from torch import nn
import torch.nn.functional as F


# 用于调整数值，使其可以被某个除数整除，常用于网络层中通道数的设置。
def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # 确保减小的百分比不超过一定的比例（round_limit）
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


# Radix Softmax用于处理分组特征的归一化
class RadixSoftmax(nn.Module):
    def __init__(self, radix, cardinality):
        super().__init__()
        self.radix = radix
        self.cardinality = cardinality

    def forward(self, x):
        batch = x.size(0)
        # 根据radix是否大于1来决定使用softmax还是sigmoid进行归一化
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = x.sigmoid()
        return x


# SplitAttn模块实现分裂注意力机制
class SplitAttn(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=None,
                 dilation=1, groups=1, bias=False, radix=2, rd_ratio=0.25, rd_channels=None, rd_divisor=8,
                 act_layer=nn.ReLU, norm_layer=None, drop_block=None, **kwargs):
        super(SplitAttn, self).__init__()
        out_channels = out_channels or in_channels
        self.radix = radix
        self.drop_block = drop_block
        mid_chs = out_channels * radix
        # 根据输入通道数、radix和rd_ratio计算注意力机制的中间层通道数
        if rd_channels is None:
            attn_chs = make_divisible(
                in_channels * radix * rd_ratio, min_value=32, divisor=rd_divisor)
        else:
            attn_chs = rd_channels * radix

        padding = kernel_size // 2 if padding is None else padding
        # 核心卷积层
        self.conv = nn.Conv2d(
            in_channels, mid_chs, kernel_size, stride, padding, dilation,
            groups=groups * radix, bias=bias, **kwargs)
        # 后续层以及RadixSoftmax
        self.bn0 = norm_layer(mid_chs) if norm_layer else nn.Identity()
        self.act0 = act_layer()
        self.fc1 = nn.Conv2d(out_channels, attn_chs, 1, groups=groups)
        self.bn1 = norm_layer(attn_chs) if norm_layer else nn.Identity()
        self.act1 = act_layer()
        self.fc2 = nn.Conv2d(attn_chs, mid_chs, 1, groups=groups)
        self.rsoftmax = RadixSoftmax(radix, groups)

    def forward(self, x):
        # 卷积和激活
        x = self.conv(x)
        x = self.bn0(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act0(x)

        # 计算分裂注意力
        B, RC, H, W = x.shape
        if self.radix > 1:
            # 对特征进行重组和聚合
            x = x.reshape((B, self.radix, RC // self.radix, H, W))
            x_gap = x.sum(dim=1)
        else:
            x_gap = x
        # 全局平均池化和两层全连接网络，应用RadixSoftmax
        x_gap = x_gap.mean(2, keepdims=True).mean(3, keepdims=True)
        x_gap = self.fc1(x_gap)
        x_gap = self.bn1(x_gap)
        x_gap = self.act1(x_gap)
        x_attn = self.fc2(x_gap)
        x_attn = self.rsoftmax(x_attn).view(B, -1, 1, 1)
        if self.radix > 1:
            out = (x * x_attn.reshape((B, self.radix, RC // self.radix, 1, 1))).sum(dim=1)
        else:
            out = x * x_attn
        return out


# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    block = SplitAttn(64)
    input = torch.rand(1, 64, 64, 64)
    output = block(input)
    print(output.shape)