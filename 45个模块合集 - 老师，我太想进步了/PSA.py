import torch
import torch.nn as nn
import torch.nn.functional as F


class PSA(nn.Module):
    def __init__(self, channel=512, reduction=4, S=4):
        super(PSA, self).__init__()
        self.S = S  # 尺度的数量

        # 定义不同尺度的卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(channel // S, channel // S, kernel_size=2 * (i + 1) + 1, padding=i + 1)
            for i in range(S)
        ])

        # 定义每个尺度对应的SE模块
        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channel // S, channel // (S * reduction), kernel_size=1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // (S * reduction), channel // S, kernel_size=1, bias=False),
                nn.Sigmoid()
            ) for i in range(S)
        ])

        self.softmax = nn.Softmax(dim=1)  # 用于归一化注意力权重

    def forward(self, x):
        b, c, h, w = x.size()

        # 将输入在通道维度上按尺度分割
        SPC_out = x.view(b, self.S, c // self.S, h, w)

        # 应用不同尺度的卷积操作
        conv_out = []
        for idx, conv in enumerate(self.convs):
            conv_out.append(conv(SPC_out[:, idx, :, :, :]))
        SPC_out = torch.stack(conv_out, dim=1)

        # 应用SE模块进行通道注意力加权
        se_out = [se(SPC_out[:, idx, :, :, :]) for idx, se in enumerate(self.se_blocks)]
        SE_out = torch.stack(se_out, dim=1)
        SE_out = SE_out.expand(-1, -1, -1, h, w)  # 扩展以匹配SPC_out的尺寸

        # 应用Softmax归一化注意力权重
        softmax_out = self.softmax(SE_out)

        # 应用注意力权重并合并多尺度特征
        PSA_out = SPC_out * softmax_out
        PSA_out = torch.sum(PSA_out, dim=1)  # 沿尺度维度合并特征

        return PSA_out


if __name__ == '__main__':
    input = torch.randn(3, 512, 64, 64)
    psa = PSA(channel=512, reduction=4, S=4)
    output = psa(input)
    print(output.shape)