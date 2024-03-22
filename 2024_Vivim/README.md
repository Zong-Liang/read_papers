# Vivim: a Video Vision Mamba for Medical Video Object Segmentation

Vivim：一种用于医学视频目标分割的视频视觉 Mamba 网络

## 针对任务

医学视频对象分割

## 动机

传统卷积难以获取全局信息，基于 Transformer 的架构的注意力机制有重大的计算负担，Mamba 通过实现一种选择性扫描机制来有效捕获一维序列交互的长距离依赖关系，同时具有线性复杂度。

U-Mamba 设计了一个混合 CNN-SSM 块，主要由 Mamba 模块组成，用于处理生物医学图像分割任务中的长序列。Vision Mamba 提供了一个新的通用视觉骨干，在图像分类和语义分割任务上使用了双向 Mamba 块。

## Method

Vivim 主要由两个模块组成：一个具有堆叠的 Temporal Mamba 块的分层编码器，用于在不同尺度提取粗糙和细粒度的特征序列，以及一个轻量级的基于 CNN 的解码器头，用于融合多级特征序列并预测分割掩码。

给定一个具有 $T$ 帧的视频片段，即 $\mathbf{V}=\{I_1,...,I_T\}$，我们首先通过重叠的块嵌入将这些帧划分为 $4×4$ 大小的块。然后，我们将块序列输入到我们的分层 Temporal Mamba 编码器中，以获得具有原始帧分辨率的 $\{1/4,1/8,1/16,1/32\}$ 的多级时空特征。最后，我们将多级特征传递给基于 CNN 的解码器头来预测分割结果。

### Overall architecture

![image-20240314140902458](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403141431423.png)

**Hierarchical feature representation：**多级特征提供了高分辨率的粗糙特征和低分辨率的细粒度特征，这显著改善了分割结果，特别是对于医学图像。为此，与 Vivit 不同，我们的编码器提取了给定输入视频帧的多级多尺度特征。具体来说，我们在每个时间 Mamba 块的末尾逐帧执行块合并，导致具有分辨率 $\begin{aligned}\frac H{2^{i+1}}\times\frac W{2^{i+1}}\end{aligned}$ 的第 $i$ 个特征嵌入 $\mathcal{F}_i$。

**Temporal Mamba Block：**首先引入了一个高效的空间自注意模块，用于提供空间信息的初始聚合，然后是一个 Mix-FeedForward 层。我们利用 SegFormer 和 Pyramid Vision Transformer 中介绍的序列缩减过程来提高其效率。对于给定视频片段的 $i$ 级特征嵌入 $\mathcal{F}_i\in\mathbb{R}^{T\times C_i\times H\times W}$，我们对通道和时间维度进行转置，并将时空特征嵌入展平为一维长序列 $h_i\in\mathbb{R}^{C_i\times THW}$。然后，将展平的序列 $h_{i}$ 输入到一个时空 Mamba 模块和一个细节特定的前馈层中。时空 Mamba 模块建立了帧内和帧间的长距离依赖关系，而细节特定的前馈层 (DSF) 通过一个核大小为 $3 × 3 × 3$ 的深度卷积来保留细粒度的细节。堆叠 Mamba 层中的过程可以定义为，其中 $l\in[1,N_m]$：

$h^l=\text{ST-Mamba}\left(\text{LN}\left(h^{l-1}\right)\right)+h^{l-1},h^l=\text{DSF}\left(\text{LN}\left(h^l\right)\right)+h^l.$

最后，我们将输出特征序列恢复到原始形状，并采用重叠的块合并来对特征嵌入进行下采样。

**Decoder：**为了从多级特征嵌入中预测分割掩码，我们引入了基于 CNN 的分割头。虽然我们的分层时间 Mamba 编码器在空间和时间轴上具有较大的有效感受野，但基于 CNN 的分割头进一步细化了局部区域的细节。具体来说，来自时间 Mamba 模块的多级特征 $\{\mathcal{F}_{1},\mathcal{F}_{2},\mathcal{F}_{3},\mathcal{F}_{4}\}$ 被传递到一个 MLP 层以统一通道维度。这些统一的特征被上采样到相同的分辨率并串联在一起。第三，采用 MLP 层来融合串联的特征 $\mathcal{F}$。最后，融合特征经过一个 $1×1$ 的卷积层来预测分割掩码 $\mathcal{M}$。在训练过程中应用由像素级交叉熵损失和 IoU 损失组成的分割损失 $\mathcal{L}_{seg}$。

### Spatiotemporal Selective Scan

尽管 S6 对于时间数据具有因果关系，但视频与文本的不同之处在于它们不仅包含时间冗余信息，还累积了非因果的二维空间信息。为了解决适应非因果数据的问题，我们引入了 ST-Mamba，它将时空序列建模纳入视频视觉任务中。

具体来说，为了明确探索帧之间的关系，我们首先将每个帧的块沿着行和列展开成序列，然后将帧序列串联起来构成视频序列 $\begin{aligned}h_i^t\in\mathbb{R}^{C_i\times T(HW)}\end{aligned}$。我们同时沿着前向和后向方向进行扫描，探索双向时间依赖关系。这种方法允许模型在不显著增加计算复杂度的情况下互相补偿各自的感受野。同时，我们沿着时间轴堆叠块，并构建空间序列 $h_i^s\in\mathbb{R}^{C_i{\times}(HW)T}$。我们继续扫描以整合来自所有帧的每个像素的信息。时空选择性扫描明确考虑了单帧空间连贯性和跨帧连贯性，并利用并行 SSM 建立帧内和帧间的长程依赖关系。具有时空选择性扫描的结构化状态空间序列模型 (ST-Mamba) 作为构建 Temporal Mamba 块的核心元素，构成了 Vivim 的基本构建模块。

**Computational-Efficiency：**ST-Mamba 中的 SSM 和 Transformer 中的自注意力都提供了自适应建模时空上下文的关键解决方案。给定一个视频视觉序列 $\mathbf{K}\in\mathbb{R}^{1\times T\times M\times D}$，全局自注意力和 SSM 的计算复杂度分别为：

$\Omega(\text{self-attention})=4(\text{TM})\text{D}^2+2(\text{TM})^2\text{D},~\Omega(\text{SSM})=4(\text{TM})(2\text{D})\text{N}+(\text{TM})(2\text{D})\text{N}^2$

其中，默认扩展比率为 $2$，$N$ 是一个固定参数，设置为 $16$。观察到，自注意力对整个视频序列长度 (TM) 是二次的，而 SSM 对其是线性的。这种计算效率使得 ST-Mamba 成为长期视频应用的更好解决方案。

### Boundary-aware Constraint

仅通过分割监督优化的网络往往会生成模糊且无结构的预测。为了缓解这个问题，我们引入了一个受 InverseForm  启发的边界感知约束，以强制预测的边界结构。具体来说，我们通过优化真值边缘和特征图中边缘之间的仿射变换到恒等变换矩阵来解决这个约束任务。在块内的真值边缘是通过在真值掩码上应用 Sobel 算子得到的，而特征补丁则通过一个辅助边界头处理。我们使用预先训练的 MLP 计算第 $i$ 个补丁之间的仿射变换矩阵 $\hat{\theta}_{i}$，该矩阵用于真值边缘和处理后的特征块之间的转换。这个 MLP 是提前用边缘掩码进行训练的，在我们方法的训练过程中不进行优化。我们通过将这个仿射变换矩阵优化为单位矩阵 ${\mathbb{I}}$ 来进行优化：

$\begin{aligned}\mathcal{L}_{affine}=\frac1{N_p}\sum_{i=1}^{N_p}\left|\hat{\theta}_i-\mathbb{I}\right|_F,\end{aligned}$

其中，$N_p$ 表示补丁的数量，$\left|\cdot\right|_{F}$ 表示 Frobenius 范数。

我们还计算整个预测边界和相应真值之间的二元交叉熵损失 $\mathcal{L}_{bce}$，以进一步优化边界检测。最后，在训练期间优化的总体损失如下，其中缩放参数 $\lambda_1,\lambda_2$ 均经验性地设置为 0.3：

$\mathcal{L}=\mathcal{L}_{seg}+\lambda_1\mathcal{L}_{affine}+\lambda_2\mathcal{L}_{bce}.$

## Experiments

### Dataset and Implementation

在两个医学视频对象分割任务上评估了 Vivim，即视频甲状腺超声分割和视频息肉分割。

**Video thyroid segmentation：**我们收集了一个视频甲状腺超声分割数据集 VTUS。VTUS 包括 100 个视频序列，每个患者一个视频序列，由三位专家标注的 9342 帧像素级真值。整个数据集按 7:3 划分为训练集和测试集，共有 70 个训练视频和 30 个测试视频。

**Video polyp segmentation：**我们采用了四个广泛使用的息肉数据集，包括基于图像的 Kvasir 和基于视频的 CVC-300、CVC-612 和 ASU-Mayo。按照 Progressively Normalized Self-Attention Network for Video Polyp Segmentation 中的相同协议，我们在 Kvasir、ASU-Mayo 以及 CVC-300 和 CVC-612 的训练集上训练我们的模型，并在测试数据集 CVC-300-TV、CVC-612-V 和 CVC-612-T 上进行三个实验。

**Implementation details：**所提出的框架是在一台 NVIDIA RTX 3090 GPU 上进行训练，并在 PyTorch 平台上实现的。我们的框架经验性地进行了 100 个 epochs 的端到端训练，使用 Adam 优化器。初始学习率设置为 $1×10^{-4}$，并衰减到 $1×10^{-6}$。在训练过程中，我们将视频帧调整为 256 × 256 的大小，并在每次迭代中将一批 4 个视频片段，每个片段包含 5 帧，输入到网络。

### Comparsion with state-of-the-art methods

**Video thyroid US segmentation：**我们采用了四个分割评估指标，包括 Dice 系数、Jaccard 系数、Precision (精确度) 和Recall (召回率)；它们的精确定义，请参考 Deep Attentional Features for Prostate Segmentation in Ultrasound。我们还通过计算每秒处理的帧数 (FPS) 来报告推理速度性能。

如表 1 所示，我们在 VTUS 数据集上定量比较了我们的方法与许多最先进的方法。这些方法包括流行的医学图像分割方法 (UNet、UNet++、TransUNet、SETR、DAF)，以及视频对象分割方法 (OSVOS、ViViT、STM、AFB-URR 、DPSTT) 。为了公平比较，我们使用它们的公开代码复现了这些方法。我们可以观察到，与图像为基础的方法相比，视频为基础的方法往往表现更好，这表明探索时间信息对于在超声视频中分割甲状腺结节具有显著优势。更重要的是，在所有基于图像和基于视频的分割方法中，我们的 Vivim 在所有评分中都取得了最高的性能，并且在 FPS 方面也是观察到的所有视频方法中运行时间最佳的。

![image-20240314151527749](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403141515881.png)

如图 2 所示，我们可视化了所选帧上的甲状腺分割结果。我们的模型能够更准确地定位和分割目标病变，并具有更准确的边界。

![image-20240314151600114](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403141516245.png)

**Video polyp segmentation：**我们采用了六个指标，遵循 Progressively Normalized Self-Attention Network for Video Polyp Segmentation 的方法，即最大 Dice (maxDice)、最大特异性 (maxSpe)、最大 IoU (maxIoU)、S-measure ($S_{\alpha}$)、E-measure ($E_{\phi}$) 和平均绝对误差 (MAE)。

我们将我们的方法与总结在表 2 中的现有方法进行比较，包括UNet、UNet++、ResUNet、ACSNet、PraNet 和 PNS-Net。我们在 CVC-300-TV、CVC-612-V  和 CVC-612-T 上进行了三个实验来验证模型的性能。在 CVC-300-TV  上，我们的 Vivim 取得了显著的性能，并且在各项指标上都明显优于所有方法 (例如，maxDice 提高了6.1%，maxIoU 提高了 8.6%) 。在 CVC-612-V 和 CVC-612-T 上，我们的 Vivim 在最大 Dice 方面一直优于其他最先进方法，尤其是分别提高了 2.4%  和 1.2%。我们还在图 3 中可视化了 CVC-612-T 上连续帧的息肉分割结果。

![image-20240314151618951](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403141516102.png)

![image-20240314152210080](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403141522189.png)

### Ablation Study

我们在 VTUS 数据集上进行了大量实验，以评估我们主要组件的有效性。为此，我们从我们的方法构建了四个基线网络。第一个基线 (标记为 “basic”) 是将我们网络中的所有 Mamba 层和边界感知约束移除。这意味着 “basic” 等于普通的 Transformer。然后，我们将具有时间正向 SSM ($T^f$) 的 ST-Mamba 层引入到“basic”中，构建另一个基线网络 “C1”，并进一步配备具有时间反向 SSM ($T^b$) 的 ST-Mamba，构建基线网络 “C2”。基于“C2”，将空间 SSM (S) 并入 ST-Mamba，构建 “C3”。因此，“C3” 等于从我们网络的训练中去除边界感知约束。表 3 报告了我们方法和四个基线网络的结果。与 “basic” 相比，“C1”在所有指标上都有相当大的提升，这表明普通的 SSM 有助于探索时间依赖性，从而提高了视频分割性能。“C2” 相对于 “C1” 的更好的 Dice 和 Jaccard 结果表明，引入我们的双向时间 SSM 可以受益于跨帧连贯性。此外，通过将 SSM 调整到非因果信息，“C3” 在 Dice 和 Recall 方面比 “C2” 提高了 0.46% 和 0.83% 的显著幅度。最后，我们的方法在 Dice、Jaccard 和 Precision方面优于 “C3”，这表明边界感知约束可以进一步帮助提升 VTUS 的结果。

![image-20240314152544601](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403141525714.png)

## Conclusion

在本文中，我们提出了一种基于 Mamba 的框架 Vivim，用于解决医学视频对象分割的挑战，特别是由于 CNN 的固有局部性和自注意机制的高计算复杂性而产生的建模长程时空依赖性的挑战。Vivim 的主要思想是将具有时空选择性扫描的结构化状态空间模型 ST-Mamba 引入到标准的分层 Transformer 架构中。这有助于以比使用自注意机制更廉价的方式探索单帧空间连贯性和跨帧连贯性。我们对我们收集的 VTUS 数据集和结肠镜检查视频进行的实验结果表明，Vivim 优于最先进的分割网络。