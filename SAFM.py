import torch
import torch.nn as nn
import torch.nn.functional as F


# 定义SAFM类，继承自nn.Module
class SAFM(nn.Module):
    def __init__(self, dim, n_levels=4):
        super().__init__()
        # n_levels表示特征会被分割成多少个不同的尺度
        self.n_levels = n_levels
        # 每个尺度的特征通道数
        chunk_dim = dim // n_levels

        # Spatial Weighting：针对每个尺度的特征，使用深度卷积进行空间加权
        self.mfr = nn.ModuleList(
            [nn.Conv2d(chunk_dim, chunk_dim, 3, 1, 1, groups=chunk_dim) for i in range(self.n_levels)])

        # Feature Aggregation：用于聚合不同尺度处理过的特征
        self.aggr = nn.Conv2d(dim, dim, 1, 1, 0)

        # Activation：使用GELU激活函数
        self.act = nn.GELU()

    def forward(self, x):
        # x的形状为(B,C,H,W)，其中B是批次大小，C是通道数，H和W是高和宽
        h, w = x.size()[-2:]

        # 将输入特征在通道维度上分割成n_levels个尺度
        xc = x.chunk(self.n_levels, dim=1)

        out = []
        for i in range(self.n_levels):
            if i > 0:
                # 计算每个尺度下采样后的大小
                p_size = (h // 2 ** i, w // 2 ** i)
                # 对特征进行自适应最大池化，降低分辨率
                s = F.adaptive_max_pool2d(xc[i], p_size)
                # 对降低分辨率的特征应用深度卷积
                s = self.mfr[i](s)
                # 使用最近邻插值将特征上采样到原始大小
                s = F.interpolate(s, size=(h, w), mode='nearest')
            else:
                # 第一尺度直接应用深度卷积，不进行下采样
                s = self.mfr[i](xc[i])
            out.append(s)

        # 将处理过的所有尺度的特征在通道维度上进行拼接
        out = torch.cat(out, dim=1)
        # 通过1x1卷积聚合拼接后的特征
        out = self.aggr(out)
        # 应用GELU激活函数并与原始输入相乘，实现特征调制
        out = self.act(out) * x
        return out


if __name__ == '__main__':
    # 创建一个SAFM实例并对一个随机输入进行处理
    x = torch.randn(1, 36, 224, 224)
    Model = SAFM(dim=36)
    out = Model(x)
    print(out.shape)