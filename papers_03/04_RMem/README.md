# RMem: Restricted Memory Banks Improve Video Object Segmentation

# RMem：限制记忆库以提升视频目标分割准确性

## 1.动机

为了应对视频目标分割中越来越具有挑战性的场景，当前的VOS算法倾向于通过扩展记忆库来容纳更多的历史信息，作者怀疑这种做法可能会因为冗余信息而降低VOS模块识别可靠特征的能力，为此作者设计了一个记忆解码的实验，证实了自己的怀疑。针对这个问题，作者提出了一种简单的方法RMem来限制记忆库的大小，通过UCB启发式的记忆更新策略和时间位置嵌入来增强时空推理，在最近的具有挑战性的数据集（包括VOST和Long Videos数据集）上达到了最优性能，验证了RMem的有效性。

## 2.方法

### 记忆解码分析（扩展记忆库如何影响 VOS 模块的解码能力）

#### 公式化

将现有框架视为基于记忆的编码器-解码器网络，编码器 $\mathbf{E}(\cdot)$ 是将视频帧 $I_{t}$ 编码为特征 $F_{t}$ 的视觉骨干，解码器 $\mathbf{D}(\cdot)$ 通过读取读取存储在内存 $\mathbf{M}[F_{0:t-1}]$ 中的特征，将特征 $F_{t}$ 转换为分割结果 ${S}_{t}$。

$$F_t=\mathbf{E}(I_t),\quad S_t=\mathbf{D}(F_t,\mathbf{M}[F_{0:t-1}])\quad(1),\quad \tilde{S}_{t}$$

VOS 的最终目标是最小化预测掩码 ${S}_{t}$和真值 $\tilde{S}_{t}$ 之间的差异。

#### 记忆解码实验设计

利用存储在记忆库中的特征解码初始帧（第0帧）的掩码。

$$S_t^0=\mathbf{D}^{\prime}(F_0,\mathbf{M}[F_{1:t}])\quad(2)$$

为什么使用上面这个公式？

(1)相关信息的存在：$\mathbf{M}[F_{1:t}])$ 中已经包含第0帧的掩码信息。

(2)同样的预测目标：对于不同的内存大小，预测目标保持不变。

(3)与常规VOS协作：使用独立的 $\mathbf{D'}(\cdot)$ 进行解码，保持原有的 VOS 过程不变，可以使用相同的内存库。

#### 实施

选择VOST数据集，采用AOT作为VOS编解码器，用与训练好的 $\mathbf{D}(\cdot)$ 初始化 $\mathbf{D'}(\cdot)$，然后用分割 loss 监督预测掩码 $S_{0}^{t}$ 和真值 $\tilde{S}_{0}$。

#### 假设与期望

通过扩展记忆库，$\mathbf{M}[F_{1:t}])$ 中的信息在后期帧中仅变得严格丰富，而预测目标不变。假设 VOS 解码器 $\mathbf{D}(\cdot)$ 能够从越来越大的 $\mathbf{M}[F_{1:t}])$ 中提取相关特征，因此，我们自然期望解码后的掩码 $S_{0}^{t}$ 在后期帧中表现出稳定或更好的精度。

#### 结果和分析

观察到掩码 $S_{0}^{t}$ 随着记忆库的增加而下降，如图 2 (b) 所示。为了验证不断增长的记忆库确实是退化的原因，我们根据经验将记忆库限制为 8 帧，其中包含最相关和最新的信息，直观地： $\mathbf{M}[F_{1:t}])$ 中的前 7 帧和最新帧。根据图 2 (b)中的蓝色曲线，限制内存只存储简洁的特征，有效地避免了退化。

![image-20240716160709941](https://github.com/Zong-Liang/ImageBed@main//202407161607037.png)

用一组简洁的相关特征限制内存库可能会通过更精确的注意力来有利于 VOS 模块的解码。

### RMem方法

![image-20240716161641450](https://github.com/Zong-Liang/ImageBed@main//202407161616538.png)

#### 限制内存库

将记忆库限制为固定帧数 K。选择最可靠的帧（帧 0）和时间最相关的帧（最近的帧）。

$$\mathrm{M}^{t+1}=\mathrm{Concat}(\mathrm{M}_0^t, \mathrm{M}_{2:K_t-1}^t, F_t)\quad(3)$$

将 $\mathrm{M}[F_{0:t-1}]$ 表示为 $\mathrm{M}_t$，$K_{t}\leq{K}$，$\mathrm{M}_{2:K_{t}-1}^{t}$ 和 $F_t$ 是最接近的帧，删除 $\mathrm{M}_1^t$。

#### 内存更新

随机选择或保留最新帧的朴素启发式方法是不可靠的，因为它们没有考虑帧的相关性或存在知识偏移。因此，我们提出了考虑相关原型特征和新鲜传入信息的原则。

如何从 K 个候选者中选择并删除最过时的帧 $K_d$ ？

利用上置信界（UCB）算法

$$O_j=R_j+\sqrt{(2\log T)/t_j}\quad(4)$$

$R_j$ 是选项 $j$ 的平均奖励，$T$ 是总的时间戳，$t_j$ 是选择 $j$ 的时间戳数量。迁移到 VOS 中时，重新定义 $R_j$ 是可靠 VOS 的帧相关性，$\sqrt{(2\log T)/t_j}$ 作为记忆的新鲜度。然后根据最小的 $O_{1:K}$ 选择要删除的帧 $K_d$。在实践中，使用 $\mathrm{M}_k^t$ 和当前 VOS 目标 $F_t$ 之间的注意力分数来定义相关项 $R_k$ 以量化来自记忆库的特征的贡献。

$$F_t^D=\mathrm{Attn}(\mathrm{Q}=F_t,\mathrm{K}=\mathrm{M}^t,\mathrm{V}=\mathrm{M}^t)\quad(5)$$

假设 $S^t$ 是在注意力内部计算的 $F_t$ 和 $M^t$ 之间的分数（经过 softmax）。然后，我们将分数之和视为记忆中帧的相关性：$R_{k} = \mathrm{sum}(S_{k}^{t})$，其中 $S_k^t$ 是对应于 $\mathrm{M}_{k}^{t}$ 的注意力分数切片。至于 UCB 中的第二项 $\sqrt{(2\log T)/t_j}$ ，我们通过定义 $t_j$ 为帧在记忆库中停留的时间，$T$ 为所有帧的总停留时间来修改它。这个新鲜度项对长时间停留的帧进行惩罚，并允许从最新信息中刷新。最后，$O_k$ 通过权重 α 将相关性项 $R_k$与之结合，平衡它们的数值比例。

#### 具有时间感知的记忆

VOS 算法通常在训练时使用短视频片段，这些片段在记忆中只有几帧，而在推理时视频则长得多。因此，如果不加限制，记忆中的帧数差异将更加显著。

引入时间位置嵌入 (TPE) 来增强时空推理，TPE 的目标是将明确的时间感知嵌入到记忆中，并指导公式 (5) 中的注意力机制。

最优记忆大小 $K$，尽管比扩展的小得多，但仍然可能大于训练时记忆大小 $Ktrain$，记忆中的帧数从 1 到 K 不等。受到 ViT 如何使用可学习的 PE 和插值来处理不同图像分辨率的启发，类似的，根据 $Ktrain$ 初始化 PE，记为 $\tilde{P}_{0:K_{\mathrm{train}}-1}$，查询帧 $F_t$ 有专用的 PE $P_q$。然后记忆库 $\mathrm{M}_{0:K_t-1}^t$的 时间 PE 为 $\mathrm{P}_{0:K_t-1}^t$。

$$P_{0:K_t-1}^t=\begin{cases}\widetilde{P}_{0:K_t-1},&K_t\leq K_\text{train}\\\text{Interp}(\widetilde{P}_{0:K_\text{train}-1},K_t),&K_t>K_\text{train}\end{cases}(6)$$

其中 $\mathrm{Interp}(\cdot)$ 通过最近邻插值将 $\tilde{P}_{0:K_{\mathrm{train}}-1}$ 插值到 $K_t$。最后，TPE 通过增加键和值来增强公式 (5) 中的原始注意力。

$$\begin{aligned}F_{t}^{D}&=\mathrm{Attn}(\mathrm{Q}=F_{t}+P_{q},\mathrm{K}=\mathrm{M}_{0:K_{t}-1}^{t}+P_{0:K_{t}-1}^{t},\mathrm{V}=\mathrm{M}_{0:K_{t}-1}^{t})&\text{(7)}\end{aligned}$$

上述设计包含两个关键选择：

(1)使用记忆中的相对索引 $\{k=0,...,K_{t}-2\}$ 而不是帧索引 t，以避免训练和推理之间的偏移。

(2)使用可学习的 PE 而不是傅里叶特征，更适合有限的训练长度 $K_{train}$。

## 3.实验

### 数据集

VOST

![image-20240716171624916](https://github.com/Zong-Liang/ImageBed@main//202407161716961.png)

Long Videos Dataset

![image-20240716171641912](https://github.com/Zong-Liang/ImageBed@main//202407161716956.png)

LVOS、YoutubeVOS、DAVIS。

### 消融实验

![image-20240716171745877](https://github.com/Zong-Liang/ImageBed@main//202407161717968.png)

![image-20240716171754998](https://github.com/Zong-Liang/ImageBed@main//202407161717035.png)

![image-20240716171833576](https://github.com/Zong-Liang/ImageBed@main//202407161718622.png)

![image-20240716171841355](https://github.com/Zong-Liang/ImageBed@main//202407161718410.png)

![image-20240716171903770](https://github.com/Zong-Liang/ImageBed@main//202407161719852.png)

![image-20240716171927193](https://github.com/Zong-Liang/ImageBed@main//202407161719232.png)

## 4.结论

通过限制记忆库的大小，可以显著提升视频目标分割（VOS）的准确性。这与现有方法倾向于扩大记忆库的做法不同，因为扩大记忆库会导致信息冗余，增加了VOS模块解码相关特征的难度。

采用限制记忆库的方法不仅提升了VOS的准确性，还减少了训练和推理过程中记忆长度的差异，从而促进了时间推理，并引入了被忽视的“时间位置嵌入”。

论文提出的“RMem”方法通过在记忆库中保持有限数量的关键帧，实现了在复杂场景下的卓越表现，并在VOST和长视频数据集上建立了新的性能基准。