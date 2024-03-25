import torch
from torch import nn
from efficientnet_pytorch.model import MemoryEfficientSwish


# 定义一个通过卷积和激活函数生成注意力图的模块
class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),  # 1x1卷积用于调整通道数
            MemoryEfficientSwish(),  # 使用MemoryEfficientSwish作为激活函数
            nn.Conv2d(dim, dim, 1, 1, 0)  # 再次1x1卷积
        )

    def forward(self, x):
        return self.act_block(x)


# 定义高效注意力机制的主体模块
class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, group_split=[4, 4], kernel_sizes=[5], window_size=4,
                 attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        # 参数初始化和定义
        assert sum(group_split) == num_heads  # 确保分组数量之和等于头的数量
        assert len(kernel_sizes) + 1 == len(group_split)  # 确保核大小列表加一等于分组数量
        self.dim = dim  # 输入通道数
        self.num_heads = num_heads  # 注意力头的数量
        self.dim_head = dim // num_heads  # 每个头的维度
        self.scalor = self.dim_head ** -0.5  # 缩放因子
        self.kernel_sizes = kernel_sizes  # 核大小列表
        self.window_size = window_size  # 窗口大小
        self.group_split = group_split  # 分组列表
        # 根据核大小和分组定义卷积层、注意力映射层和QKV层
        convs = []
        act_blocks = []
        qkvs = []
        for i in range(len(kernel_sizes)):
            kernel_size = kernel_sizes[i]
            group_head = group_split[i]
            if group_head == 0:
                continue
            convs.append(nn.Conv2d(3 * self.dim_head * group_head, 3 * self.dim_head * group_head, kernel_size,
                                   1, kernel_size // 2, groups=3 * self.dim_head * group_head))
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(nn.Conv2d(dim, 3 * group_head * self.dim_head, 1, 1, 0, bias=qkv_bias))
        if group_split[-1] != 0:
            # 对最后一个全局注意力头的定义
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        # 将定义的模块注册为子模块
        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)  # 输出投影层
        self.attn_drop = nn.Dropout(attn_drop)  # 注意力dropout
        self.proj_drop = nn.Dropout(proj_drop)  # 投影dropout

    # 高频注意力处理函数
    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()
        qkv = to_qkv(x)  # (b (3 m d) h w)
        qkv = mixer(qkv).reshape(b, 3, -1, h, w).transpose(0, 1).contiguous()  # (3 b (m d) h w)
        q, k, v = qkv  # (b (m d) h w)
        attn = attn_block(q.mul(k)).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        res = attn.mul(v)  # (b (m d) h w)
        return res

    # 低频注意力处理函数
    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        '''
        x: (b c h w)
        '''
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()  # (b m (h w) d)
        kv = avgpool(x)  # (b c h w)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()  # (2 b m (H W) d)
        k, v = kv  # (b m (H W) d)
        attn = self.scalor * q @ k.transpose(-1, -2)  # (b m (h w) (H W))
        attn = self.attn_drop(attn.softmax(dim=-1))
        res = attn @ v  # (b m (h w) d)
        res = res.transpose(2, 3).reshape(b, -1, h, w).contiguous()
        return res

    # 模块的前向传播
    def forward(self, x: torch.Tensor):
        '''
        x: (b c h w)
        '''
        res = []
        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i]))
        if self.group_split[-1] != 0:
            res.append(self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool))
        return self.proj_drop(self.proj(torch.cat(res, dim=1)))


# 输入 N C HW,  输出 N C H W
if __name__ == '__main__':
    block = EfficientAttention(64).cuda()
    input = torch.rand(1, 64, 64, 64).cuda()
    output = block(input)
    print(output.shape)