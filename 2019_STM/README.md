## 基本信息

- 标题：Video Object Segmentation using Space-Time Memory Networks
- 作者：Seoung Wug Oh
- 机构：Yonsei University (延世大学)
- 发表时间：2019 年
- 出处：ICCV 2019
- 链接：https://github.com/seoungwugoh/STM

## 方法

### Space-Time Memory Networks (STM)

视频帧是使用第一帧中给出的真值从第二帧开始顺序处理的。在视频处理过程中，将带有对象掩码的过去帧 (在第一帧给出或在其他帧估计) 视为记忆帧，将没有对象掩码的当前帧视为查询帧。

记忆帧和查询帧首先通过专用的深度编码器编码为键和值映射对。请注意，查询编码器仅将图像作为输入，而记忆编码器同时接受图像和对象掩码。每个编码器输出 Key 和 Value 图。键用于寻址。具体来说，计算查询帧和记忆帧的关键特征之间的相似性，以确定何时何地从中检索相关的记忆值。因此，学习键对视觉语义进行编码，以匹配对外观变化具有鲁棒性。另一方面，值存储详细信息以生成掩码估计 (例如目标对象和对象边界)。来自查询和内存的值包含出于某种不同目的的信息。具体来说，学习查询帧的值来存储详细的外观信息，以便我们解码准确的对象掩码。学习记忆帧的值来编码视觉语义和关于每个特征是否属于前景或背景的掩码信息。

键和值进一步通过时空记忆读取块。查询和记忆帧的关键特征图上的每个像素在视频的时空空间上密集匹配。然后使用相对匹配分数来定位记忆帧的值特征图，并将相应的值组合起来返回输出。最后，解码器获取读取块的输出，并为查询帧重建掩码。

![image-20240611214729382](https://cdn.jsdelivr.net/gh/Zong-Liang/ImageBed@main//202406112147494.png)

### Key and Value Embedding

**查询编码器：**查询编码器将查询帧作为输入。编码器通过附加到骨干网络的两个并行卷积层输出两个特征图——键和值。这些卷积层用作瓶颈层，以减少骨干网络输出的特征通道大小（键为 1/8，值为 1/2），并且没有应用非线性。查询嵌入的输出是一对二维键和值映射($\mathrm{k}^Q\in\mathbb{R}^{{H}\times W\times C/8},\mathrm{v}^Q\in\mathbb{R}^{H\times W\times C/2}$)，其中 H 为高度，W 为宽度，C 为骨干网输出特征图的特征维数。

```python
class Encoder_Q(nn.Module):
    def __init__(self):
        super(Encoder_Q, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
```

**记忆编码器：**除了输入之外，记忆编码器具有相同的结构。记忆编码器的输入由 RGB 帧和对象掩码组成。对象掩码表示为 0 到 1 之间的单通道概率图（softmax 输出用于估计的掩码）。在输入内存编码器之前，输入沿通道维度连接。如果有多个内存帧，则它们中的每一个都独立嵌入到键和值映射中。然后，来自不同内存帧的键和值映射沿时间维度堆叠。内存嵌入的输出是一对 3D 键和值映射 ($\mathrm{k}^M\in\mathbb{R}^{T\times H\times W\times C/8},\mathrm{v}^M\in\mathbb{R}^{T\times H\times W\times C/2}$)，其中 T 是内存帧的数量。将 ResNet50 作为内存和查询编码器的骨干网络。我们使用 ResNet50 的 stage-4 (res4) 特征图作为基础特征图来计算键和值特征图。对于记忆编码器，通过植入额外的单通道过滤器来修改第一个卷积层以便能够采用 4 通道张量。网络权重是从 ImageNet 预训练模型初始化的，除了新添加的随机初始化过滤器。

```python
class Encoder_M(nn.Module):
    def __init__(self):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.resnet50(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1 # 1/4, 256
        self.res3 = resnet.layer2 # 1/8, 512
        self.res4 = resnet.layer3 # 1/8, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std
        m = torch.unsqueeze(in_m, dim=1).float() # add channel dim
        o = torch.unsqueeze(in_o, dim=1).float() # add channel dim

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_o(o) 
        x = self.bn1(x)
        c1 = self.relu(x)   # 1/2, 64
        x = self.maxpool(c1)  # 1/4, 64
        r2 = self.res2(x)   # 1/4, 256
        r3 = self.res3(r2) # 1/8, 512
        r4 = self.res4(r3) # 1/8, 1024
        return r4, r3, r2, c1, f
```

### Space-time Memory Read

在内存读取操作中，首先通过测量查询键图和内存键图所有像素之间的相似性来计算软权重。通过将内存键映射中的每个时空位置与查询键映射中的每个空间位置进行比较，以非局部方式执行相似度匹配。然后，通过带有软权重的加权求和来检索内存的值，并将其与查询值连接。此内存读取对查询特征图上的每个位置进行操作，可以概括为：$\mathbf{y}_i=\begin{bmatrix}\mathbf{v}_i^Q,&\frac1Z\sum_{\forall j}f(\mathbf{k}_i^Q,\mathbf{k}_j^M)\mathbf{v}_j^M\end{bmatrix},$ 其中 i 和 j 是查询和内存位置的索引，$Z=\sum_{\forall j}f(\mathbf{k}_{i}^{Q},\mathbf{k}_{j}^{M})$ 是归一化因子，$[·, ·]$ 表示连接。相似度函数 f 如下：$f(\mathbf{k}_i^Q,\mathbf{k}_j^M)=\exp(\mathbf{k}_i^Q\circ\mathbf{k}_j^M),$ 其中$\circ$表示点积。我们的公式可以看作是差分记忆网络的早期公式扩展到用于视频像素匹配的 3D 时空空间的扩展。因此，所提出的读取操作定位内存的时空位置以进行检索。它还与非局部自注意力机制有关，因为它执行非局部匹配。然而，我们的公式的动机是出于不同的目的，因为它旨在关注其他（记忆帧）进行信息检索，而不是本身用于自注意力。如图 3 所示，我们的内存读取操作可以通过现代深度学习平台中基本张量操作的组合轻松实现。

![image-20240611222430466](https://cdn.jsdelivr.net/gh/Zong-Liang/ImageBed@main//202406112224540.png)

### Decoder

解码器接受读取操作的输出并重建当前帧的对象掩码。我们使用[24]中使用的细化模块作为我们解码器的构建块。读取输出首先通过卷积层和残差块压缩为 256 个通道，然后许多细化模块一次逐渐将压缩的特征图放大 2 倍。每个阶段的细化模块通过跳跃连接将前一阶段的输出和来自查询编码器的特征图从相应尺度上获取。最后一个细化块的输出用于通过最终的卷积层重建对象掩码，然后是 softmax 操作。解码器中的每个卷积层都使用 3×3 过滤器，产生 256 通道输出，除了最后一个产生 2 通道输出的卷积层。解码器在输入图像的1/4尺度上估计掩码。

```python
class Decoder(nn.Module):
    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(1024, mdim, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim) # 1/8 -> 1/4
        self.RF2 = Refine(256, mdim) # 1/4 -> 1

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3,3), padding=(1,1), stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4) # out: 1/8, 256
        m2 = self.RF2(r2, m3) # out: 1/4, 256

        p2 = self.pred2(F.relu(m2))
        
        p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p #, p2, p3, p4
```

### Multi-object Segmentation

我们框架的描述基于视频中有一个目标对象。然而，最近的基准测试需要一种可以处理多对象的方法。为了满足这一要求，我们使用掩码合并操作扩展了我们的框架。我们独立地为每个对象运行我们的模型，并计算所有对象的掩码概率图。然后，我们使用类似于[24]的软聚合操作来合并预测映射。在[24]中，掩码合并仅在测试期间作为后处理步骤执行。在这项工作中，我们将操作首次使用为差分网络层，并在训练和测试期间应用它。补充材料中提供了更多细节。

### Two-stage Training

我们的网络首先在从静态图像数据生成的模拟数据集上进行预训练。然后，通过主要训练进一步针对真实世界的视频进行微调。

**图像的预训练：**我们框架的一个优势是它不需要长时间的训练视频。这是因为该方法在没有任何时间平滑度假设的情况下学习远距离像素之间的语义时空匹配。这意味着我们可以用几个帧1和对象掩码来训练我们的网络。这使我们能够使用图像数据集模拟训练视频。以前的一些工作[26，24]使用静态图像训练了他们的网络，我们采用了类似的策略。通过将随机仿射变换2应用于具有不同参数的静态图像，生成由3帧组成的合成视频剪辑。我们利用用对象掩码(显著目标检测-[29,5]、语义分割-[7,8,19])注释的图像数据集来预训练我们的网络。通过这样做，我们可以期望我们的模型对各种对象外观和类别具有鲁棒性。

**视频的主要训练：**在预训练之后，我们使用真实的视频数据进行主要训练。在这一步中，使用 Youtube-VOS 或 DAVIS-2017，具体取决于目标评估基准。为了使训练样本，我们从训练视频中采样 3 个时间有序的帧。为了学习长时间外观变化，我们在采样期间随机跳过帧。在训练过程中，要跳过的最大帧数从 0 逐渐增加到25，如课程学习。

**使用动态记忆进行训练：**在训练期间，内存使用网络的先前输出动态更新。随着系统逐帧向前移动，将上一步计算的分割输出添加到下一帧的内存中。没有阈值的原始网络输出，即前景对象的概率图，直接用于内存嵌入来模拟估计的不确定性。

**训练细节：**我们使用随机裁剪的 384×384 块进行训练。对于所有实验，我们将 minibatch 大小设置为 4 并禁用所有批量归一化层。我们使用固定学习率为 1e-5 的 Adam 优化器 [15] 最小化交叉熵损失。预训练大约需要 4 天，主要训练大约需要 3 天，使用四个 NVIDIA GeForce 1080 Ti GPU。