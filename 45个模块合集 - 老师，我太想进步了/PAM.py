import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_mask(input_size, window_size, inner_size, device):
    """Get the attention mask of PAM-Naive"""
    # Get the size of all layers
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)

    # 12 + 6 + 3 + 1 = 22
    seq_length = sum(all_size)
    mask = torch.zeros(seq_length, seq_length, device=device)

    # get intra-scale mask
    inner_window = inner_size // 2
    for layer_idx in range(len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = max(i - inner_window, start)
            right_side = min(i + inner_window + 1, start + all_size[layer_idx])
            # mask[i, left_side:right_side] = 1
            if layer_idx == 0:
                mask[i, 0:right_side] = 1
            else:
                mask[i, left_side:right_side] = 1

    # get inter-scale mask
    for layer_idx in range(1, len(all_size)):
        start = sum(all_size[:layer_idx])
        for i in range(start, start + all_size[layer_idx]):
            left_side = (start - all_size[layer_idx - 1]) + (i - start) * window_size[layer_idx - 1]
            if i == (start + all_size[layer_idx] - 1):
                right_side = start
            else:
                right_side = (start - all_size[layer_idx - 1]) + (i - start + 1) * window_size[layer_idx - 1]
            mask[i, left_side:right_side] = 1
            mask[left_side:right_side, i] = 1

    # mask = (1 - mask).bool()
    mask = mask.bool()

    return mask, all_size


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv2d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=[1, window_size],
                                  stride=[1, window_size])
        torch.nn.init.xavier_uniform_(self.downConv.weight)

    def forward(self, x):
        # (B,T,N,D)-> (B, D, N, T)
        x = x.permute(0, 3, 2, 1)
        x = self.downConv(x)
        x = F.relu_(x)
        return x.permute(0, 3, 2, 1)  # (B, D, N, T)-->(B,T,N,D)


class Conv_Construct(nn.Module):
    """Convolution CSCM"""

    def __init__(self, d_model, window_size):
        super(Conv_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size),
                ConvLayer(d_model, window_size)
            ])
        else:
            self.conv_layers = nn.ModuleList([
                ConvLayer(d_model, window_size[0]),
                ConvLayer(d_model, window_size[1]),
                ConvLayer(d_model, window_size[2])
            ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        # enc_input: (B, T, N, D)
        all_inputs = []
        all_inputs.append(enc_input)  # 先把初始输入放进列表内

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](
                enc_input)  # 以卷积核[2,2,2]为例: 1th conv: (B,T,N,D)-->(B,T/2,N,D); 2th conv: (B,T/2,N,D)-->(B,T/4,N,D); 3th conv: (B,T/4,N,D)-->(B,T/8,N,D)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs,
                               dim=1)  # 在时间维度上拼接多尺度的序列: (B,T,N,D) + (B,T/2,N,D) + (B,T/4,N,D) + (B,T/8,N,D) = (B,M,N,D);  令M=T+T/2+T/4+T/8
        all_inputs = self.norm(all_inputs)

        return all_inputs


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, dropout, normalize_before, all_size):
        super().__init__()

        D = n_head * d_k
        self.all_size = all_size
        self.d = d_k
        self.K = n_head
        self.mask = True
        self.FC_q = nn.Linear(D, D)
        self.FC_k = nn.Linear(D, D)
        self.FC_v = nn.Linear(D, D)
        self.FC = nn.Linear(D, D)

    def forward(self, q, k, v, mask=None):
        batch_size_ = q.shape[0]

        query = self.FC_q(q)  # 生成q矩阵: (B,M,N,D)--> (B,M,N,D)
        key = self.FC_k(k)  # 生成k矩阵: (B,M,N,D)--> (B,M,N,D)
        value = self.FC_v(v)  # 生成v矩阵: (B,M,N,D)--> (B,M,N,D)

        query = torch.cat(torch.split(query, self.d, dim=-1), dim=0)  # 划分为多头: (B,M,N,D)-->(B*k,M,N,d);  D=k*d
        key = torch.cat(torch.split(key, self.d, dim=-1), dim=0)  # 划分为多头: (B,M,N,D)-->(B*k,M,N,d);  D=k*d
        value = torch.cat(torch.split(value, self.d, dim=-1), dim=0)  # 划分为多头:  (B,M,N,D)-->(B*k,M,N,d);  D=k*d

        query = query.permute(0, 2, 1, 3)  # 进行变换,为了便于计算: (B*k,M,N,d)-->(B*k,N,M,d)
        key = key.permute(0, 2, 3, 1)  # (B*k,M,N,d)-->(B*k,N,d,M)
        value = value.permute(0, 2, 1, 3)  # (B*k,M,N,d)-->(B*k,N,M,d)

        attention = torch.matmul(query, key)  # 得到注意力矩阵: (B*k,N,M,d) @ (B*k,N,d,M) = (B*k,N,M,M)
        attention /= (self.d ** 0.5)

        # 屏蔽掉注意力矩阵中那些没有连接的节点对
        if self.mask:
            num = torch.tensor(-2 ** 15 + 1)
            num = num.to(torch.float32).to(device)
            attention = torch.where(mask, attention,
                                    num)  # 如果mask某元素是fasle,那么attention矩阵的对应位置应填入负无穷数值(即num),这样的话在执行softmax之后负无穷对应的位置应当趋近于0
        # softmax
        attention = F.softmax(attention, dim=-1)

        # [batch_size, num_step, num_vertex, D]
        X = torch.matmul(attention, value)  # 通过注意力矩阵聚合对应节点的信息: (B*k,N,M,M) @ (B*k,N,M,d) = (B*k,N,M,d)
        X = X.permute(0, 2, 1, 3)  # (B*k,N,M,d)-->(B*k,M,N,d)
        X = torch.cat(torch.split(X, batch_size_, dim=0),
                      dim=-1)  # 在通道上拼接多头注意力的输出::(B*k,M,N,d)-->(B,M,N,k*d)==(B,M,N,D)
        X = self.FC(X)  # 通过映射层融合多个子空间的特征
        return X[:, :self.all_size[0]]  # 只选择原始序列的长度进行输出: (B,M,N,D)-->(B,T,N,D)


if __name__ == '__main__':
    # (B,T,N,D)  N:序列的个数, T:时间序列的长度;   注意: 输入长度是8,window_size=[2,2,2]; 如果输入长度是12,window_size=[2,2,3]; 确保除到最后一层长度为1: 8/2/2/2=1; 12/2/2/3=1;
    X = torch.randn(1, 8, 1, 64)
    seq_length = X.shape[1]

    # 得到金字塔的mask矩阵; 以输入序列长度等于8,三层卷积核分别为[2, 2, 2]为例子: all_size=[8,4,2,1],存放每一个尺度对应的序列长度
    mask, all_size = get_mask(input_size=seq_length, window_size=[2, 2, 2], inner_size=3, device=device)
    # 通过卷积构造金字塔结构  (粗尺度构造模块)
    conv_layers = Conv_Construct(d_model=64, window_size=[2, 2, 2])
    # 定义多头注意力机制
    Model = MultiHeadAttention(n_head=8, d_model=64, d_k=8, dropout=0., normalize_before=False, all_size=all_size)

    X = conv_layers(X)  # 执行粗尺度构造模块(B,T,N,D)-->(B,M,N,D)
    output = Model(X, X, X, mask=mask)  # 执行注意力机制： (B,M,N,D)--> (B,T,N,D)
    print(output.shape)