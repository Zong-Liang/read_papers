# Mamba-UNet: UNet-Like Pure Visual Mamba for Medical Image Segmentation

Mamba-UNet: 用于医学图像分割的类 UNet 架构的纯视觉 Mamba

## Mamba-UNet架构

![image-20240313144652260](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403131447980.png)

### Architecture Overview

首先将尺寸为 $H×W×1$ 的输入的二维灰度图像分成与 VMamba 一样的 patch，然后转换为尺寸为 $\frac{H}{4}\times\frac{W}{4}\times16$ 的一维序列。初始的线性嵌入层将特征维度调整为任意大小 $C$。然后，这些 patch tokens 通过多个 VSS 块和块合并层进行处理，创建分层特征。块合并层用来下采样和增加维度，而 VSS 块专注于学习特征表示。编码器的每个阶段的输出分辨率分别为 $\frac{H}{4}\times\frac{W}{4}\times C$、$\frac{H}{8}\times\frac{W}{8}\times 2C$、$\frac{H}{16}\times\frac{W}{16}\times 4C$ 和 $\frac{H}{32}\times\frac{W}{32}\times 8C$。解码器包括 VSS 块和块扩展层遵循编码器的样式，使输出具有完全相同的特征大小，从而可以通过跳跃连接来增强由于下采样而丢失的空间细节。在编码器和解码器中，分别使用 2 个 VSS 块，并加载预训练的 VMamba-Tiny 模型到编码器中，按照 Swin-UNet 加载预训练的 SwinViT-Tiny 模型的相同过程进行操作。

### VSS Block

在 VSS 块中，输入特征首先经过线性嵌入层，然后分成两个路径。一个分支经过深度卷积和 SiLU 激活，然后进入 SS2D 模块，并在后层归一化之后与另一支经过 SiLU 激活后的特征进行合并。这个 VSS 块不使用位置嵌入，不像典型的视觉 Transformer，而是选择了一个新的结构，没有 MLP 阶段，这使得在相同的深度预算内可以实现更密集的块堆叠。

![image-20240313150437478](C:/Users/ZL/AppData/Roaming/Typora/typora-user-images/image-20240313150437478.png)

### Encoder

在编码器中，分辨率降低的 $C$ 维的 token 化的输入经过两个连续的 VSS 块进行特征学习，保持维度和分辨率不变，在 Mamba-UNet 的编码器中，将作为下采样过程的块合并操作利用了三次，通过将输入分割成 4 个象限并连接，然后每次通过归一化层标准化维度，将 token 数量减少了一半，同时将特征维度增加了一倍。

### Decoder

与编码器相对应，解码器利用了两个连续的 VSS 块进行特征重建，使用补丁扩展层来放大深层特征。这些层提高了分辨率（2× 放大），同时将特征维度减少了一半，例如，在重新组织和减少特征维度以提高分辨率之前，首先通过一个初始层来使特征维度加倍。

### Bottleneck & Skip Connetions

Mamba-UNet 的瓶颈部分利用了两个 VSS 块。每一层的编码器和解码器都使用跳跃连接将多尺度特征与放大的输出融合在一起，通过合并浅层和深层特征来增强空间细节。随后的线性层保持了这个综合特征集的维度，确保与放大后的分辨率一致。

## Experiments and Results

### Data Sets

来自 MICCAI 2017 挑战赛的公开可用的 ACDC MRI 心脏分割数据集。

### Baseline Methods

与 UNet 和 Swin-UNet 直接进行比较。

### Evaluation Metrics

$\begin{aligned}Dice&=\frac{2\times TP}{2\times TP+FP+FN}\end{aligned}$

$\begin{aligned}\text{Accuracy}&=\frac{TP+TN}{TP+TN+FP+FN}\end{aligned}$

$\begin{aligned}\text{Precision}=\frac{TP}{TP+FP}\end{aligned}$

$\begin{aligned}\text{Sensitivity}=\frac{TP}{TP+FN}\end{aligned}$

$\begin{aligned}\mathrm{Specificity}=\frac{TN}{TN+FP}\end{aligned}$

其中，$TP$ 代表真阳性的数量，$TN$ 代表真阴性的数量，$FP$ 表示假阳性的数量，$FN$ 表示假阴性的数量。

$\begin{aligned}\text{Hausdorff Distance (HD) 95\%}=\max\left(\max_{a\in A}\min_{b\in B}d(a,b),\max_{b\in B}\min_{a\in A}d(a,b)\right)_{95\%}\end{aligned}$

$\begin{aligned}\text{Average Surface Distance (ASD)}=\frac1{|A|+|B|}\left(\sum_{a\in A}\min_{b\in B}d(a,b)+\sum_{b\in B}\min_{a\in A}d(a,b)\right)\end{aligned}$

其中，$a$ 和 $b$ 分别表示预测和真值上的点集。$d(a, b)$ 表示两点之间的欧氏距离。$95%$ 是 $Hausdorff$ 距离的修改版本，着重于距离的第 $95$ 个百分位数，以减少异常值的影响。

### Qualitative Results

![image-20240313172555068](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403131725228.png)

### Quantitative Results

![image-20240313172658288](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403131726376.png)

![image-20240313172802859](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403131728933.png)

## Conclusion

在本文中，我们介绍了 Mamba-UNet，这是一种纯粹基于 Visual Mamba 块的 UNet 风格网络，用于医学图像分割。性能表明，Mamba-UNet 在性能上优于经典的类似网络，如 UNet 和 Swin-UNet。未来，我们的目标是对来自不同模态和目标的更多医学图像分割任务进行更深入的探索，并与更多的分割骨干进行比较。此外，我们计划将Mamba-UNet 扩展到 3D 医学图像，并采用半监督/弱监督学习来进一步促进医学成像的发展。
