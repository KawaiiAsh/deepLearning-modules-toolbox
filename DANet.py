import numpy as np
import torch
from torch import nn
from torch.nn import init


# 双重注意力网络（DANet）用于场景分割任务的实现


class ScaledDotProductAttention(nn.Module):
    '''
    实现缩放点积注意力机制。
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1):
        '''
                参数:
                :param d_model: 模型的输出维度
                :param d_k: 查询和键的维度
                :param d_v: 值的维度
                :param h: 头的数量
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        前向传播函数
        参数:
        :param queries: 查询 (b_s, nq, d_model)
        :param keys: 键 (b_s, nk, d_model)
        :param values: 值 (b_s, nk, d_model)
        :param attention_mask: 注意力遮罩 (b_s, h, nq, nk)，掩码为True的地方将被忽略
        :param attention_weights: 注意力权重 (b_s, h, nq, nk)
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SimplifiedScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SimplifiedScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = queries.view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = keys.view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = values.view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class PositionAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = ScaledDotProductAttention(d_model, d_k=d_model, d_v=d_model, h=1)

    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.shape
        y = self.cnn(x)  # (B, C, H, W) --> (B, C, H, W)
        y = y.view(B, C, -1).permute(0, 2, 1)  # (B, C, H, W) --> (B,C,N)-->(B,N,C)   N=H*W
        y = self.pa(y, y, y)  # (B,N,C)
        return y


class ChannelAttentionModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        self.cnn = nn.Conv2d(d_model, d_model, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.pa = SimplifiedScaledDotProductAttention(H * W, h=1)

    def forward(self, x):
        # (B, C, H, W)
        B, C, H, W = x.shape
        y = self.cnn(x)  # (B, C, H, W) --> (B, C, H, W)
        y = y.view(B, C, -1)  # (B, C, H, W)-->(B, C, N)  N=H*W
        y = self.pa(y, y, y)  # (B, C, N)
        return y


class DAModule(nn.Module):

    def __init__(self, d_model=512, kernel_size=3, H=7, W=7):
        super().__init__()
        # 位置注意力和通道注意力的区别就是：通道注意力没有通过卷积操作生成qkv
        self.position_attention_module = PositionAttentionModule(d_model=512, kernel_size=3, H=7, W=7)
        self.channel_attention_module = ChannelAttentionModule(d_model=512, kernel_size=3, H=7, W=7)

    def forward(self, input):
        # (B, C, H, W)
        B, C, H, W = input.shape
        p_out = self.position_attention_module(input)  # 执行位置注意力: (B, C, H, W)-->(B,N,C)
        c_out = self.channel_attention_module(input)  # 执行通道注意力:(B, C, H, W)--> (B, C, N)
        p_out = p_out.permute(0, 2, 1).view(B, C, H, W)  # (B,N,C)-->(B,C,N)-->(B,C,H,W)
        c_out = c_out.view(B, C, H, W)  # (B,C,N)-->(B,C,H,W)
        return p_out + c_out


# 两个注意力机制就不细讲了哦, 基本一模一样,只不过通道注意力没有通过卷积生成新的qkv,作者说会破坏原有通道之间的相关性。
if __name__ == '__main__':
    # (B, C, H, W)
    input = torch.randn(1, 512, 7, 7)
    Model = DAModule(d_model=512, kernel_size=3, H=7, W=7)
    output = Model(input)
    print(output.shape)