import torch
import torch.nn as nn
from thop import profile  # 引入thop库来计算模型的FLOPs和参数数量


# 定义SimAM模块
class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.Sigmoid()  # 使用Sigmoid激活函数
        self.e_lambda = e_lambda  # 定义平滑项e_lambda，防止分母为0

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入x的尺寸
        n = w * h - 1  # 计算特征图的元素数量减一，用于下面的归一化
        # 计算输入特征x与其均值之差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算注意力权重y，这里实现了SimAM的核心计算公式
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 返回经过注意力加权的输入特征
        return x * self.act(y)


# 示例使用
if __name__ == '__main__':
    model = Simam_module().cuda()  # 实例化SimAM模块并移到GPU上
    x = torch.randn(1, 3, 64, 64).cuda()  # 创建一个随机输入并移到GPU上
    y = model(x)  # 将输入传递给模型
    print(y.size())  # 打印输出尺寸
    # 使用thop库计算模型的FLOPs和参数数量
    flops, params = profile(model, (x,))
    print(flops / 1e9)  # 打印以Giga FLOPs为单位的浮点操作数
    print(params)  # 打印模型参数数量