import numpy as np
import torch
from torch import nn
from torch.nn import init


# EMSA模块，为多尺度注意力机制的实现
class EMSA(nn.Module):

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, H=7, W=7, ratio=3, apply_transform=True):

        super(EMSA, self).__init__()
        # 初始化参数和层
        self.H = H  # 输入特征图的高度
        self.W = W  # 输入特征图的宽度
        self.fc_q = nn.Linear(d_model, h * d_k)  # 查询向量的全连接层
        self.fc_k = nn.Linear(d_model, h * d_k)  # 键向量的全连接层
        self.fc_v = nn.Linear(d_model, h * d_v)  # 值向量的全连接层
        self.fc_o = nn.Linear(h * d_v, d_model)  # 输出的全连接层
        self.dropout = nn.Dropout(dropout)  # Dropout层

        self.ratio = ratio  # 空间降采样比例
        if (self.ratio > 1):
            # 如果空间降采样比例大于1，则添加空间降采样层
            self.sr = nn.Sequential()
            self.sr_conv = nn.Conv2d(d_model, d_model, kernel_size=ratio + 1, stride=ratio, padding=ratio // 2,
                                     groups=d_model)
            self.sr_ln = nn.LayerNorm(d_model)

        self.apply_transform = apply_transform and h > 1
        if (self.apply_transform):
            # 如果应用变换，则添加变换层，包括1x1卷积、Softmax和实例归一化
            self.transform = nn.Sequential()
            self.transform.add_module('conv', nn.Conv2d(h, h, kernel_size=1, stride=1))
            self.transform.add_module('softmax', nn.Softmax(-1))
            self.transform.add_module('in', nn.InstanceNorm2d(h))

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

        b_s, nq, c = queries.shape
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)

        if (self.ratio > 1):
            x = queries.permute(0, 2, 1).view(b_s, c, self.H, self.W)  # bs,c,H,W
            x = self.sr_conv(x)  # bs,c,h,w
            x = x.contiguous().view(b_s, c, -1).permute(0, 2, 1)  # bs,n',c
            x = self.sr_ln(x)
            k = self.fc_k(x).view(b_s, -1, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, n')
            v = self.fc_v(x).view(b_s, -1, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, n', d_v)
        else:
            k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
            v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        if (self.apply_transform):
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = self.transform(att)  # (b_s, h, nq, n')
        else:
            att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, n')
            att = torch.softmax(att, -1)  # (b_s, h, nq, n')

        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


if __name__ == '__main__':
    block = EMSA(d_model=512, d_k=512, d_v=512, h=8, H=8, W=8, ratio=2, apply_transform=True).cuda()
    input = torch.rand(64, 64, 512).cuda()
    output = block(input, input, input)
    print(output.shape)