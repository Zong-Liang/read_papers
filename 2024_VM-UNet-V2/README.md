# VM-UNET-V2: Rethinking Vision Mamba UNet for Medical Image Segmentation

VM-UNET-V2：重新思考用于医学图像分割的 Vision Mamba UNet

## Methods

### VM-UNetV2 Architecture

![image-20240319130009540](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403191300561.png)

Vision Mamba UNetV2 包含三个主要模块：编码器、SDI 语义和细节融合模块、解码器。给定一个图像输入 $I$，其中 $R^{H\times W\times3}$，编码器生成 $M$ 个级别的特征。我们将第 $i$ 个级别的特征表示为 $f_i^o$，其中 $1≤i≤M$。这些累积的特征 $\{f_{1}^{o},f_{2}^{o},...,f_{M}^{o}\}$ 随后被送到 SDI 模块进行进一步的增强。如图所示，$f_i$ 的编码器输出通道为 $2^{i}\times C$，$\{f_{1}^{o},f_{2}^{o},...,f_{M}^{o}\}$ 共同进入 SDI 模块进行特征融合，$f_i$ 对应于 $f_i'$ 作为第 $i$ 个阶段的输出。$f_i$ 的特征尺寸是 $\frac H{2^{i+1}}\times\frac{\tilde{W}}{2^{i+1}}\times2^{i}C$。在我们的模型中，我们使用深度监督来计算 $f_i'$ 和 $f_{i-1}'$ 的特征的损失。

在本文中，我们在编码器四个阶段采用 $[N_1,N_2,N_3,N_4]$ 个 VSS 块，每个阶段的通道数为 $[C,2C,4C,8C]$。通过观察 VMamba，我们发现 $N_3$ 和 $c$ 的不同值是区分 Tiny、Small 和 Base 框架规范的重要因素。 按照 VMamba 的规范，我们让 $C$ 取 96、$N_1$ 和 $N_2$ 都取 2，$N_3$ 从集合 $[2,9,27]$ 中取值。这表示我们使用 VMamba 的 Tiny 和 Small模型作为消融实验的主干。 

### VSS And SDI Block

![image-20240319130036668](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403191300812.png)

来自 VMamba 的 VSS 块作为 VMUNetV2 编码器的主干。VSS 块的结构如图所示，首先通过初始线性嵌入层进行处理，然后将其划分为两个独立的信息流，一个通过 $3×3$ 深度卷积层进行定向，然后在进入主要 SS2D 模块前进行 SiLU 函数激活，然后 SS2D 的输出通过层归一化后和另外一个经过 SiLU 函数激活的信息流相结合，结合后的信息组成了 VSS 块的最终输出。

使用编码器生产的分层特征图 $f_i^o = \frac H{2^{i+1}}\times\frac{\tilde{W}}{2^{i+1}}\times2^{i}C$，其中 $i$ 表示第 $i$ 个级别，$1≤i≤4$。

SDI 模块可以使用不同的注意力机制来计算空间和通道的注意力分数。按照 UNetV2 中提到的内容，我们使用CBAM 来实现空间和时间注意力。计算公式 $f_i^1=\phi_i^{att}\left(f_i^0\right)$，$\phi_i^{att}$ 表示第 $i$ 个注意力的计算。然后我们使用 $1×1$ 的卷积将 $f_i^1$ 的通道与 $c$ 对齐，得到的特征图为 $f_{i}^{2}\in R^{H_{i}\times W_{i}\times c}$。

在 SDI 解码器的第 $i$ 个阶段，$f_i^2$ 表示参考目标。然后我们调整每个第 $j$ 层的特征图的大小以匹配 $f_i^2$ 的大小，公式如下：

$\left.f_{ij}^3=\left\{\begin{array}{ll}\mathrm{G_d}\left(f_j^2,(H_i,W_i)\right)&\mathrm{if~}j<i,\\\mathrm{G_I}\left(f_j^2\right)&\mathrm{if~}j=i,\\\mathrm{G_U}\left(f_j^2,(H_i,W_i)\right)&\mathrm{if~}j>i,\end{array}\right.\right.$

其中，$G_d$、$G_i$ 和 $G_u$ 分别表示自适应平均池化、身份映射和双线性插值。

$\begin{aligned}f_{ij}^4&=\theta_{ij}\left(f_{ij}^3\right)\\f_i^5&=H\left([f_{i1}^4,f_{i2}^4,f_{i3}^4,f_{i4}^4]\right)\end{aligned}$

其中，$\theta_{ij}$ 代表平滑卷积的参数，$f_{ij}^4$ 是第 $i$ 个级别的第 $j$ 个平滑特征图。这里 $H()$ 表示 hadamard 积。随后哦，在第 $i$ 层将 $f_i^5$ 送到解码器，以进一步重建分辨率和分割。

### Loss function

对于医学图像分割任务，主要使用基本的 CrossEntropy 和 Dice 损失作为损失函数，导致所有的数据集掩码都包含两个类，它们是单个目标和背景。

$\begin{aligned}
&L_{\mathrm{BceDice}}=\lambda_{1}L_{\mathrm{Bce}}+\lambda_{2}L_{\mathrm{Dice}} \\
&L_{Bce}=-\frac{1}{N}\sum_{1}^{N}\left[y_{i}log\left(\hat{y}_{i}\right)+\left(1-y_{i}\right)log\left(1-\hat{y}_{i}\right)\right] \\
&\begin{aligned}L_{\mathrm{Dice}}=1-\frac{2|X\cap Y|}{|X|+|Y|}\end{aligned}
\end{aligned}$

$(\lambda_{1},\lambda_{2})$ 是常数，通常默认为 $(1,1)$。

## Experiments and results

### Datasets

我们使用三种类型的数据集来验证我们框架的有效性。第一种类型是开源皮肤疾病数据集，包括 ISIC 2017 和 ISIC 2018，我们以 7: 3 的比例将皮肤数据集拆分为训练集和测试集。第二个是开源胃肠道息肉数据集，其中包括 Kvasir-SEG、ClinicDB、ColonDB、Endoscene 和 ETIS，在这种类型的数据集中，我们遵循 PraNet 中的实验设置。对于这些数据集，我们对几个指标进行了详细的评估，包括 Mean Intersection over Union(mIoU)、Dice Similarity Coefficient(DSC)、Accuracy(Acc)、Sensitivity(Sen) 和 Specificity(Spe)。

### Experimental setup

在 VMamba 工作之后，我们将所有数据集中的图像维度调整为 256 × 256 像素。为了抑制过度拟合，我们还带来了数据增强方法，例如随机翻转和随机旋转。在操作参数方面，我们有 80 的批大小，AdamW 优化器以 1e-3 的学习率开始。我们使用 CosineAnnealingLR 作为调度程序，其操作最多跨越 50 次迭代，学习率低至 1e-5。我们在 300 个 epoch 的过程中进行训练。对于 VM-UNetV2，编码器单元的权重最初设置为与 VMamba-S 的权重对齐。该实现是在 Ubuntu 20.04 系统上进行的，使用 Python3.9.1、PyTorch2.0.1 和 CUDA11.7，所有实验均在单个 NVIDIA RTX V100 GPU 上进行。

### Results

我们将 VM-UNetV2 与一些最先进的模型进行比较，结果如表 1 和表 2 所示。对于 ISIC 数据集，我们的 VM-UNetV2 在 mIoU、DSC 和 Acc 指标方面优于其他模型。在息肉相关数据集中，我们的模型在所有指标上都超过了最先进的模型 UNetV2，mIoU 参数增加了高达 7%。

![image-20240319135704435](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403191400812.png)

![image-20240319135719934](C:/Users/ZL/AppData/Roaming/Typora/typora-user-images/image-20240319135719934.png)

除了评估模型的准确性之外，我们还评估了不同模型的计算复杂度，利用了 VMamba 的线性复杂度的优势。我们使用模型推理速度 FPS（每秒帧数）、模型参数计数 (Params) 和用于评估的浮点运算 (FLOPs) 的数量等指标，如表 3 所示。 (3, 256, 256) 表示输入图像的大小。所有测试均在 NVIDIA V100 GPU 上进行。VM-UNetV2 的 FLOPs 和FPS 优于其他的 FLOPs 和 FPS。

![image-20240319135809911](C:/Users/ZL/AppData/Roaming/Typora/typora-user-images/image-20240319135809911.png)

### Ablation studies

在本节中，我们使用息肉数据集对 VMUNetV2 编码器的初始化和解码器的深度监督操作进行了消融实验。如VMamba 论文所述，编码器的深度和特征图中的通道数决定了 VMamba 的规模。在本文中，所提出的 VM-UNetV2 仅使用 ImageNet-1k 上 VMamba 的预训练权重作为编码器部分。因此，在本研究中进行模型尺度消融实验时，我们只改变编码器的深度，如表 4 所示。对于输出特征，我们采用深度监督机制，利用两层输出特征的融合，然后将其与真实标签进行比较，进行损失计算。如表 4 和表 5 所示，当编码器的深度设置为 [2, 2, 9, 2] 时，分割评估指标相对较好。因此，当使用 VM-UNetV2 时，不需要选择特别大的深度。在大多数使用深度监督机制的情况下，分割评估指标相对较好，但它不是一个决定性因素。对于不同的数据集，需要分别进行消融实验，以确定是否采用深度监督机制。

![image-20240319140017655](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403191400804.png)

## Conclusions

在本文中，我们提出了一种基于 SSM 的UNet 类型医学图像分割模型 VM-UNetV2，该模型充分利用了基于 SSM 的模型的能力。我们使用 VSS 块和 SDI 分别处理 Encoder 和 Skip 连接。VMamba 的预训练权重用于初始化 VM-UNetV2 的编码器部分，并采用深度监督机制监督多个输出特征。我们的模型已在皮肤病和息肉数据集上进行了测试。结果表明，我们的模型在分割任务中具有很强的竞争力。复杂性分析表明，VM-UNetV2 在 FLOPs、Params 和 FPS 方面也很有效。