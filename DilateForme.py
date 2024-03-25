import numpy as np
import torch
from torch import flatten, nn
from torch.nn import init
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import functional as F

"DilateFormer: Multi-Scale Dilated Transformer for Visual Recognition"


class DilateAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, head_dim, qk_scale=None, attn_drop=0, kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        # nn.unfold(): Extracts sliding local blocks from a batched input tensor.  local block是根据卷积核、膨胀率确定下来的，以此达到局部、稀疏的效果
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=dilation * (kernel_size - 1) // 2,
                                stride=1)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        # (B, scale_d, H, W)    scale_d == d;   C = scale_num * scale_d
        B, d, H, W = q.shape

        # 首先对q进行变换:(B,d,H,W) -->reshape--> (B,h,hd,1,HW) -->permute--> (B,h,HW,1,hd)   d=h*hd; (hd:head_dim), h:注意力头的个数, 这里的注意力头的个数和MultiDilatelocalAttention中的设置是不一样的,在MultiDilatelocalAttention中, 也就是76行, 因为N个尺度平分了C个通道,所以在这里注意力头的个数h=(C/N)/head_dim
        q = q.reshape([B, d // self.head_dim, self.head_dim, 1, H * W]).permute(0, 1, 4, 3, 2)  # B,h,N,1,d

        # 通过滑动窗口选取keys patch: (B,d,H,W) --> (B,d*k*k,HW)   作者通过zero padding填充, 使输入前后的patch数量都为HW个;  k*k是滑窗的尺寸, 经过滑动窗口移动完之后,得到HW个patch,每个patch的大小是k*k  //////  unfold的计算方式见官网：https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html#torch.nn.Unfold, 解释见博客:https://blog.csdn.net/qq_37937847/article/details/115663343
        k = self.unfold(k)
        # 对k进行变换,便于与q进行乘法: (B,d*k*k,HW) --> (B,h,hd,k*k,HW) --> (B,h,HW,hd,k*k)    d=h*hd;  HW*(k*K): HW个patch,每个patch的大小是k*K个像素,每个像素的通道是hd
        k = k.reshape([B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1,
                                                                                                                  4, 2,
                                                                                                                  3)

        # 每个query patch对选择的k*k个keys patch做注意力: (B,h,HW,1,hd) @ (B,h,HW,hd,k*k) = (B,h,HW,1,k*k)
        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 同样,通过滑动窗口选取values patch:(B,d,H,W) --> (B,d*k*k,HW) --> (B,h,hd,k*k,HW) --> (B,h,HW,k*k,hd)
        v = self.unfold(v).reshape(
            [B, d // self.head_dim, self.head_dim, self.kernel_size * self.kernel_size, H * W]).permute(0, 1, 4, 3, 2)

        # 通过attn权重对values加权求和:  (B,h,HW,1,k*k) @ (B,h,HW,k*k,hd) = (B,h,HW,1,hd) -->transpose--> (B,HW,h,1,hd) -->reshape--> (B,H,W,h*hd)==(B,H,W,d)
        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    "Implementation of Dilate-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., kernel_size=3, dilation=[1, 3]):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.scale = qk_scale or head_dim ** -0.5
        self.num_dilation = len(dilation)
        # 膨胀率的个数要能够被注意力头的个数整除才行
        assert num_heads % self.num_dilation == 0, f"num_heads{num_heads} must be the times of num_dilation{self.num_dilation}!!"
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)
        self.dilate_attention = nn.ModuleList(
            [DilateAttention(head_dim, qk_scale, attn_drop, kernel_size, dilation[i])
             for i in range(self.num_dilation)])
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape

        x = x.permute(0, 3, 1, 2)  # (B,H,W,C)-->(B,C,H,W)

        # 通过卷积层,将通道C映射到3C,并将其在通道上平分为qkv: (B,C,H,W) --> (B,3C,H,W) --> (B,3,scale_num,scale_d,H,W) --> (scale_num,3,B,scale_d,H,W)  //////  num_dilation==scale_num;  C=scale_num * scale_d   有多少个膨胀率就说明有多少个尺度,所以这里定义：num_dilation=scale_num
        qkv = self.qkv(x).reshape(B, 3, self.num_dilation, C // self.num_dilation, H, W).permute(2, 1, 0, 3, 4, 5)

        # (B,C,H,W) --> (B,scale_num,scale_d,H,W) --> (scale_num,B,H,W,scale_d)
        x = x.reshape(B, self.num_dilation, C // self.num_dilation, H, W).permute(1, 0, 3, 4, 2)

        # 循环计算每一个尺度,在这里dilation=[1, 3],有两种取值,意味着有两个尺度  【假设有8个注意力头,两个尺度, 那每一个尺度在计算注意力的时候都分成4个头, 大家可以详细的看看注释, 加油！】
        for i in range(self.num_dilation):
            # 执行dilate_attention: q=(B,scale_d,H,W)  k=(B,scale_d,H,W)  v=(B,scale_d,H,W)  输出: x[i]=(B,H,W,scale_d)
            x[i] = self.dilate_attention[i](qkv[i][0], qkv[i][1], qkv[i][2])

        # (scale_num,B,H,W,scale_d) --> (B,H,W,scale_num,scale_d) --> (B,H,W,scale_num*scale_d)==(B,H,W,C)
        x = x.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    # (B,H,W,C)
    input = torch.randn(1, 16, 16, 512)
    Model = MultiDilatelocalAttention(dim=512)
    output = Model(input)
    print(output.shape)