import torch
from torch import nn


class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            moduleation(bool, optional): If True, Modulated Defromable Convolution(Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        # self.p_conv偏置层，学习公式（2）中的偏移量。
        # 2*kernel_size*kernel_size：代表了卷积核中所有元素的偏移坐标，因为同时存在x和y的偏移，故要乘以2。
        self.p_conv = nn.Conv2d(inc, 2 * kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            # self.m_conv权重学习层，是后来提出的第二个版本的卷积也就是公式（3）描述的卷积。
            # kernel_size*kernel_size：代表了卷积核中每个元素的权重。
            self.m_conv = nn.Conv2d(inc, kernel_size * kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            # register_backward_hook是为了方便查看这几层学出来的结果，对网络结构无影响。
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    # 生成卷积核的邻域坐标
    def _get_p_n(self, N, dtype):
        """
        torch.meshgrid():Creates grids of coordinates specified by the 1D inputs in attr:tensors.
        功能是生成网格，可以用于生成坐标。
        函数输入两个数据类型相同的一维张量，两个输出张量的行数为第一个输入张量的元素个数，
        列数为第二个输入张量的元素个数，当两个输入张量数据类型不同或维度不是一维时会报错。

        其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；
        第二个输出张量填充第二个输入张量中的元素各列元素相同。
        """
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1),
            torch.arange(-(self.kernel_size - 1) // 2, (self.kernel_size - 1) // 2 + 1))

        # p_n ===>offsets_x(kernel_size*kernel_size,) concat offsets_y(kernel_size*kernel_size,)
        #     ===> (2*kernel_size*kernel_size,)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        # （1， 2*kernel_size*kernel_size, 1, 1）
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    # 获取卷积核在feature map上所有对应的中心坐标，也就是p0
    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # (b, 2*kernel_size, h, w)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    # 将获取的相对坐标信息与中心坐标相加就获得了卷积核的所有坐标。
    # 再加上之前学习得到的offset后，就是加上了偏移量后的坐标信息。
    # 即对应论文中公式(2)中的(p0+pn+Δpn)
    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        # p_n ===> (1, 2*kernel_size*kernel_size, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # p_0 ===> (1, 2*kernel_size*kernel_size, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        # (1, 2*kernel_size*kernel_size, h, w)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        # b, h, w, 2*kerel_size*kernel_size
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # x ===> (b, c, h*w)
        x = x.contiguous().view(b, c, -1)
        # 因为x是与h轴方向平行，y是与w轴方向平行。故将2D卷积核投影到1D上，位移公式如下：
        # 各个卷积核中心坐标及邻域坐标的索引 offsets_x * w + offsets_y
        # (b, h, w, kernel_size*kernel_size)
        index = q[..., :N] * padded_w + q[..., N:]
        # (b, c, h, w, kernel_size*kernel_size) ===> (b, c, h*w*kernel_size*kernel_size)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)
        # (b, c, h*w)
        # x_offset[0][0][0] = x[0][0][index[0][0][0]]
        # index[i][j][k]的值应该是一一对应着输入x的(h*w)的坐标，且在之前将index[i][j][k]的值clamp在[0, h]及[0, w]范围里。
        # (b, c, h, w, kernel_size*kernel_size)
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        # (b, c, h, w, kernel_size*kernel_size)
        b, c, h, w, N = x_offset.size()
        # (b, c, h, w*kernel_size)
        x_offset = torch.cat([x_offset[..., s:s + ks].contiguous().view(b, c, h, w * ks) for s in range(0, N, ks)],
                             dim=-1)
        # (b, c, h*kernel_size, w*kernel_size)
        x_offset = x_offset.contiguous().view(b, c, h * ks, w * ks)

        return x_offset

    def forward(self, x):
        # (b, c, h, w) ===> (b, 2*kernel_size*kernel_size, h, w)
        offset = self.p_conv(x)
        if self.modulation:
            # (b, c, h, w) ===> (b, kernel_size*kernel_size, h, w)
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        # kernel_size*kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)
        # (b, 2*kernel_size*kernel_size, h, w)
        p = self._get_p(offset, dtype)
        # (b, h, w, 2*kernel_size*kernel_size)
        p = p.contiguous().permute(0, 2, 3, 1)
        # 将p从tensor的前向计算中取出来，并向下取整得到左上角坐标q_lt。
        q_lt = p.detach().floor()
        # 将p向上再取整，得到右下角坐标q_rb。
        q_rb = q_lt + 1

        # 学习的偏移量是float类型，需要用双线性插值的方法去推算相应的值。
        # 同时防止偏移量太大，超出feature map，故需要torch.clamp来约束。
        # Clamps all elements in input into the range [ min, max ].
        # torch.clamp(a, min=-0.5, max=0.5)

        # p左上角x方向的偏移量不超过h,y方向的偏移量不超过w。
        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2) - 1), torch.clamp(q_lt[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # p右下角x方向的偏移量不超过h,y方向的偏移量不超过w。
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2) - 1), torch.clamp(q_rb[..., N:], 0, x.size(3) - 1)],
                         dim=-1).long()
        # p左上角的x方向的偏移量和右下角y方向的偏移量组合起来，得到p左下角的值。
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        # p右下角的x方向的偏移量和左上角y方向的偏移量组合起来，得到p右上角的值。
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p。
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2) - 1), torch.clamp(p[..., N:], 0, x.size(3) - 1)], dim=-1)

        # 双线性插值公式里的四个系数。即bilinear kernel。
        # 作者代码为了保持整齐，每行的变量计算形式一样，所以计算需要做一点对应变量的对应变化。
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # 计算双线性插值的四个坐标对应的像素值。
        # (b, c, h, w, kernel_size*kernel_size)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # 双线性插值的最后计算
        # (b, c, h, w, kernel_size*kernel_size)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt +
                   g_rb.unsqueeze(dim=1) * x_q_rb +
                   g_lb.unsqueeze(dim=1) * x_q_lb +
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            # (b, kernel_size*kernel_size, h, w) ===> (b, h, w, kernel_size*kernel_size)
            m = m.contiguous().permute(0, 2, 3, 1)
            # (b, h, w, kernel_size*kernel_size) ===>  (b, 1, h, w, kernel_size*kernel_size)
            m = m.unsqueeze(dim=1)
            # (b, c, h, w, kernel_size*kernel_size)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m
        # x_offset: (b, c, h, w, kernel_size*kernel_size)
        # x_offset: (b, c, h*kernel_size, w*kernel_size)
        x_offset = self._reshape_x_offset(x_offset, ks)
        # out: (b, c, h, w)
        out = self.conv(x_offset)

        return out