import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import einops


# 定义高效加性注意力模块
class EfficientAdditiveAttnetion(nn.Module):
    """
    高效加性注意力模块，用于SwiftFormer中。
    输入：形状为[B, N, D]的张量
    输出：形状为[B, N, D]的张量
    """

    def __init__(self, in_dims=512, token_dim=512):
        super().__init__()
        # 初始化查询和键的线性变换
        self.to_query = nn.Linear(in_dims, token_dim)
        self.to_key = nn.Linear(in_dims, token_dim)

        # 初始化可学习的权重向量和缩放因子
        self.w_a = nn.Parameter(torch.randn(token_dim, 1))
        self.scale_factor = token_dim ** -0.5

        # 初始化后续的线性变换
        self.Proj = nn.Linear(token_dim, token_dim)
        self.final = nn.Linear(token_dim, token_dim)

    def forward(self, x):
        B, N, D = x.shape  # B:批次大小，N:序列长度，D:特征维度

        # 生成初步的查询和键矩阵
        query = self.to_query(x)
        key = self.to_key(x)

        # 对查询和键进行标准化处理
        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        # 学习查询的注意力权重，并进行缩放和标准化
        query_weight = query @ self.w_a
        A = query_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)

        # 通过注意力权重对查询进行加权，以生成全局查询向量
        q = torch.sum(A * query, dim=1)
        q = q.reshape(B, 1, -1)

        # 计算全局查询向量和每个键的交互，再与原始查询进行逐元素相加
        out = self.Proj(q * key) + query
        out = self.final(out)  # 通过最终的线性层输出调制后的特征

        return out


if __name__ == '__main__':
    # 示例：对一个形状为[B, N, D]的随机张量应用高效加性注意力模块
    X = torch.randn(1, 50, 512)
    Model = EfficientAdditiveAttnetion(in_dims=512, token_dim=512)
    out = Model(X)
    print(out.shape)  # 输出的形状应为[B, N, D]