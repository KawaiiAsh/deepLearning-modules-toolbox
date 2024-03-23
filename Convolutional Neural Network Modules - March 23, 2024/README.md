# CNN卷积神经网络常见模块

| 模块             | 代码 | 论文                                                                                                                                                             |
|----------------|----|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| SENet          | ✅  | [论文](https://arxiv.org/pdf/1709.01507)                                                                                                                         |
| SKNet          | ✅  | [论文](https://arxiv.org/pdf/1903.06586)                                                                                                                         |
| scSE           | ✅  | [论文](http://arxiv.org/pdf/1803.02579v2)                                                                                                                        |
| Non-Local Net  | ✅  | [论文](https://arxiv.org/pdf/1711.07971)                                                                                                                         |
| GCNet          | ✅  | [论文](https://arxiv.org/abs/1904.11492)                                                                                                                         |
| CBAM           | ✅  | [论文](https://arxiv.org/abs/1807.06521)                                                                                                                         |
| BAM            | ✅  | [论文](https://arxiv.org/abs/1807.06514)                                                                                                                         |
| SplitAttention | ✅  | [论文](https://hangzhang.org/files/resnest.pdf)                                                                                                                  |
| ACNet          | ✅  | [论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_ACNet_Strengthening_the_Kernel_Skeletons_for_Powerful_CNN_via_Asymmetric_ICCV_2019_paper.pdf) |
| ASPP           | ✅  | [论文](https://arxiv.org/pdf/1802.02611)                                                                                                                         |
| BlazeBlock     | ✅  | [论文](https://www.arxiv.org/pdf/1907.05047)                                                                                                                     |
| PPM            | ✅  | [论文](https://arxiv.org/abs/1612.01105)                                                                                                                         |
| Strip Pooling  | ✅  | [论文](https://arxiv.org/abs/2003.13328v1)                                                                                                                       |

即插即用模块一般是作为一个独立的模块，可以用于取代普通的卷积结构，或者直接插入网络结构中。

最常见的即插即用模块莫过于注意力模块了，近些年好多略显水的工作都用到了注意力模块，仅仅需要简单添加这些注意力模块即可作为论文的创新点，比如SENet+Darknet53组合。

添加这类即插即用模块还需要注意几个问题：

插入的位置：有的模块适合插入在浅层，有的模块适合在深层。具体插在哪里最好看原作者论文中的插入位置作为参考。一般情况可以插入的常见位置有：

瓶颈层：比如ResNet，DenseNet的瓶颈层。
上采样层：比如FPN分支，Attention UNet。
骨干网络最后一层：比如SPP, ASPP等

所有的3x3卷积：比如深度可分离卷积等

插入后进行实验为何不生效？指标没有提高甚至降低？

很多模块虽然说是即插即用，但是并不是无脑插入以后结果就一定会提高。

注意力模块通常情况下都需要调参才能维持原本的准确率，在调参效果比较好的情况下才能超过原本的模型。

## SENet

说明：最经典的通道注意力模块，曾夺最后一节ImageNet冠军。

## SKNet

说明：SENet改进版，增加了多个分支，每个分支感受野不同。

## scSE

说明：scSE分为两个模块，一个是sSE和cSE模块，分别是空间注意力和通道注意力，最终以相加的方式融合。论文中只将其使用在分割模型中，在很多图像分割比赛中都有用到这个模块作为trick。

## Non-Local Net

说明：NLNet主要借鉴了传统方法中的非局部均值滤波设计了Non-Local全局注意力，虽然效果好，但是计算量偏大，建议不要在底层网络使用，可以适当在高层网络中使用。

## GCNet

说明：GCNet主要针对Non-Local 计算量过大的问题结合了提出了解决方案

## CBAM

说明：将空间注意力机制和通道注意力机制进行串联

## BAM

说明：和CBAM同一个作者，将通道注意力和空间注意力用并联的方式连接

## SplitAttention

说明：ResNeSt = SENet + SKNet + ResNeXt

## ACNet

说明：通过在训练过程中引入1x3 conv和3x1 conv，强化特征提取，实现效果提升

## ASPP

说明：ASPP是DeepLabv3+其中一个核心创新点，用空间金字塔池化模块来进一步提取多尺度信息，这里是采用不同rate的空洞卷积来实现这一点。

## BlazeBlock

说明：来自BlazeFace的一个模块，主要作用是轻量化

## PPM

说明：跟ASPP类似，只不过PSPNet的PPM是使用了池化进行的融合特征金字塔，聚合不同区域的上下文信息。

## Strip Pooling

说明：跟CCNet挺像的，就是对SPP这种传统的Spatial Pooling进行了改进，设计了新的体系结构。