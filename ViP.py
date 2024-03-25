import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        # 第一层全连接层
        self.fc1 = nn.Linear(in_features, hidden_features)
        # 激活函数
        self.act = act_layer()
        # 第二层全连接层
        self.fc2 = nn.Linear(hidden_features, out_features)
        # Dropout层
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # 顺序通过第一层全连接层、激活函数、Dropout、第二层全连接层、Dropout
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WeightedPermuteMLP(nn.Module):
    def __init__(self, dim, seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        # 分段维度，用于在特定维度上分段处理特征
        self.seg_dim = seg_dim

        # 定义对通道C、高度H、宽度W的MLP处理层
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_h = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim, dim, bias=qkv_bias)

        # 重置权重的MLP层
        self.reweighting = MLP(dim, dim // 4, dim * 3)

        # 最终投影层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        # 通道维度的处理
        c_embed = self.mlp_c(x)

        # 高度维度的处理
        S = C // self.seg_dim
        h_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 2, 1, 4).reshape(B, self.seg_dim, W, H * S)
        h_embed = self.mlp_h(h_embed).reshape(B, self.seg_dim, W, H, S).permute(0, 3, 2, 1, 4).reshape(B, H, W, C)

        # 宽度维度的处理
        w_embed = x.reshape(B, H, W, self.seg_dim, S).permute(0, 3, 1, 2, 4).reshape(B, self.seg_dim, H, W * S)
        w_embed = self.mlp_w(w_embed).reshape(B, self.seg_dim, H, W, S).permute(0, 2, 3, 1, 4).reshape(B, H, W, C)

        # 计算三个维度的权重并应用softmax进行归一化
        weight = (c_embed + h_embed + w_embed).permute(0, 3, 1, 2).flatten(2).mean(2)
        weight = self.reweighting(weight).reshape(B, C, 3).permute(2, 0, 1).softmax(0).unsqueeze(2).unsqueeze(2)

        # 加权融合处理后的特征
        x = c_embed * weight[0] + w_embed * weight[1] + h_embed * weight[2]

        # 应用投影层和Dropout
        x = self.proj_drop(self.proj(x))

        return x


if __name__ == '__main__':
    input = torch.randn(64, 8, 8, 512)  # 模拟输入数据
    seg_dim = 8  # 定义分段维度
    vip = WeightedPermuteMLP(512, seg_dim)  # 初始化模型
    out = vip(input)  # 前向传播
    print(out.shape)