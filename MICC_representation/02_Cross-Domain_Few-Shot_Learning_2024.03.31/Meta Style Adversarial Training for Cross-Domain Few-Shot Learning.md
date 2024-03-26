# StyleAdv: Meta Style Adversarial Training for Cross-Domain Few-Shot Learning

StyleAdv：用于跨域小样本学习的原风格对抗训练

## Abstract

跨域小样本学习 (CD-FSL) 是最近一项新兴的任务，它解决了不同领域的小样本学习问题。它旨在将在源数据集中学到的先验知识转移到新的目标数据集。CD-FSL 任务尤其受到不同数据集的巨大领域差距的挑战。至关重要的是，这种领域差距实际上来自于视觉风格的变化，Wave-SAN 经验表明，跨越源数据的风格分布有助于缓解这个问题。然而，Wave-SAN 只是简单地交换两幅图像的风格。这样的普通操作使生成的样式“真实”和“简单”仍然属于源样式的原始集合。因此，受 vanilla 对抗性学习的启发，我们提出了一种用于 CD-FSL 的新型与模型无关的元风格对抗训练 (StyleAdv) 方法以及一种新颖的风格对抗性攻击方法。特别是，我们的风格攻击方法综合了“虚拟”和“硬”对抗性风格进行模型训练。这是通过用有符号的风格梯度扰动原始风格来实现的。通过不断攻击风格并迫使模型识别这些具有挑战性的对抗性风格，我们的模型逐渐对视觉风格具有鲁棒性，从而提高了新目标数据集的泛化能力。除了典型的基于 CNN 的主干之外，我们还在大规模预训练视觉转换器上使用我们的 StyleAdv 方法。在八个不同目标数据集上进行的大量实验证明了我们方法的有效性。无论是建立在 ResNet 还是 ViT 的基础上，我们实现了 CD-FSL 的最新技术。

## Introduction

本文研究了跨域少样本学习 (CD-FSL) 的任务，该任务旨在解决了不同领域的小样本学习 (FSL) 问题。作为 FSL 的一般方法，基于情节的元学习策略也被用于训练 CD-FSL 模型，例如，FWT、LRP、ATA 和 Wave-SAN。一般来说，为了在测试阶段模拟低样本状态，元学习采样情节来训练模型。每一个情节包含一个小的有标记的支持集和一个无标记的查询集。模型通过根据支持集预测查询集中包含的图像类别来学习元知识。学习到的元知识直接将模型泛化到新的目标类。

根据经验，我们发现源数据和目标数据之间视觉外观的变化是导致 CD-FSL 中域差异的关键原因之一。有趣的是，WaveSAN，我们之前的工作表明可以通过增强源图像的视觉风格来缓解域差异问题。特别是，Wave-SAN 建议以自适应实例归一化 (AdaIN) 的形式增加样式，方法是随机采样两个源情节并交换它们的风格。然而，尽管 Wave-SAN 很有效，但这种 näıve 风格的生成方法存在两个局限性：1) 交换操作使样式始终限制在源数据集的“真实”风格集中；2) 有限的真实风格进一步导致生成的风格过于“容易”而无法学习。因此，一个自然的问题是我们是否可以合成“虚拟”和“硬”风格来学习更强大的 CD-FSL 模型？形式上，我们使用“真实/虚拟”来指示风格是否最初出现在源风格集中，并将“简单/困难”定义为新风格是否使元任务更加困难。

为此，我们从对抗训练中汲取灵感，提出了一种新的 CD-FSL 元风格对抗训练方法 (StyleAdv)。StyleAdv 在元训练的两个迭代优化循环中玩极小极大游戏。特别是，内循环通过添加扰动从原始源风格生成对抗性风格。合成的对抗性风格对于当前的模型来说应该更具挑战性，从而增加了损失。虽然外循环通过最小化识别具有原始风格和对抗性风格的图像的损失来优化整个网络。我们的最终目标是使模型能够学习一个除了源数据相对有限和简单的风格之外，对各种风格具有鲁棒性的模型。这可以潜在地提高具有视觉外观变化的新目标域的泛化能力。

形式上，我们引入了一种新的风格对抗性攻击方法来支持 StyleAdv 的内部循环。与之前攻击方法不同，我们的风格攻击方法扰动和合成风格，而不是图像像素或特征。从技术上讲，我们首先从输入特征图中提取风格，并在前向计算链中包含提取的风格，以获得每个训练步骤的梯度。之后，我们通过将一定比例的梯度添加到原始风格来合成新的风格。我们的风格对抗性攻击方法合成的风格具有“硬”和“虚拟”的良好属性。特别是，由于我们以训练梯度的相反方向扰动风格，我们的生成会导致“硬”风格。我们的攻击方法产生了与原始源风格完全不同的完全“虚拟”风格。

关键的是，我们的风格攻击方法使用变化的风格扰动比使得风格合成是渐进式的，这使得它与普通对抗性攻击方法有很大不同。具体来说，我们提出了一种新颖的渐进式风格合成策略。直接插入扰动的 näıve 解决方案是单独攻击特征嵌入模块的每个块，然而，这可能会导致特征与高级块的偏差很大。因此，我们的策略是使当前块的合成信号由先前块的对抗性风格累积。另一方面，我们不是通过固定攻击率来攻击模型，而是通过从候选池中随机采样扰动率来合成新的风格。这有助于合成对抗风格的多样性。实验结果表明了我们方法的有效性：1) 我们的风格对抗性攻击方法确实合成了更具挑战性的风格，从而推动了源视觉分布的限制；2) 我们的 StyleAdv 显着提高了基础模型并优于所有其他 CD-FSL 竞争对手。

我们强调我们的 StyleAdv 与其他现有的 FSL 或 CD-FSL 模型无关和互补，例如 GNN 和 FWT。更重要的是，为了从大规模预训练模型 (例如 DINO)中受益，我们进一步探索了调整我们的 StyleAdv 以非参数方式改进 Vision Transformer (ViT)  主干。在实验中，我们表明 StyleAdv 不仅改进了基于 CNN 的 FSL/CD-FSL 方法，而且改进了大规模预训练的 ViT 模型。

最后，我们总结了我们的贡献。1) 针对 CD-FSL 提出了一种新的元风格对抗训练方法 StyleAdv。StyleAdv 通过先扰动原始风格，然后强制模型学习这种对抗性风格，提高了 CD-FSL 模型的鲁棒性。2) 我们提出了一种新的基于变化的攻击率的渐进合成策略的风格攻击方法，因此生成了不同的“虚拟”和“硬”风格。。3) 我们的方法与现有的 FSL 和 CD-FSL 方法是互补的；我们在基于 CNN 和基于 ViT 的主干上验证了我们的想法。4) 在 8 个未见的目标数据集的广泛结果表明，我们的 StyleAdv 优于以前的 CD-FSL 方法，构建了一个新的 SOTA 结果。

## Related Work

**Cross-Domain Few-Shot Learning：**旨在将模型从对海量标注的数据的依赖中解放出来的小样本学习已经研究了许多年。特别是，最近的一些工作，例如 CLIP、CoOp、CLIP-Adapter、Tip-Adapter 和 PMF 探索了用大规模预训练模型促进 FSL。特别是，PMF 贡献了一个简单的管道，并为 FSL 构建了一个 SOTA。作为FSL的扩展任务，CD-FSL 主要解决跨不同领域的 FSL。典型的基于元学习的 CD-FSL 方法包括 FWT、LRP、ATA、AFA 和 Wave-SAN。具体来说，FWT 和 LRP 通过细化批量归一化层并使用解释模型来指导训练来解决 CD-FSL。ATA、AFA 和 Wave-SAN 分别建议增强图像像素、特征和视觉风格。还探索了几种基于迁移学习的 CD-FSL 方法，例如 BSCD-FSL (也称为微调)、BSR 和 NSAE。这些方法表明微调有助于提高目标数据集的性能。引入额外数据或需要多个域数据集进行训练的其他工作包括 STARTUP、Meta-FDMixup、Me-D2N、TGDM、TriAE 和 DSL。

**Adversarial Attack：**对抗性攻击旨在通过向输入数据添加一些定制的扰动来误导模型。为了有效地产生扰动，人们提出了许多对抗性攻击方法。大多数工作攻击图像像素。具体来说，FGSM 和 PGD 是两个最经典和最著名的攻击算法。一些工作攻击特征空间。关键的是，很少有工作攻击风格。与旨在误导模型的这些工作不同，我们扰动风格来解决 CD-FSL 的视觉偏移问题。

**Adversarial Few-Shot Learning：**已经进行了几次探索 FSL 对抗学习的尝试。其中，MDAT、AQ 和MetaAdv 首先攻击输入图像，然后利用攻击图像训练模型，提高对抗样本的防御能力。Shen等人攻击情节的特征，提高 FSL 模型的泛化能力。请注意，ATA 和 AFA，两种 CD-FSL 方法也采用了对抗性学习。但是，我们与他们有很大的不同。ATA 和 AFA 扰动图像像素或特征，而我们的目标是通过生成不同的硬风格来弥合视觉差距。

**Style Augmentation for Domain Shift Problem：**在域生成、图像分割、人员重识别和 CD-FSL 中探索了增强风格分布以缩小域偏移问题。具体来说，MixStyle、AdvStyle、DSU 和 Wave-SAN 通过混合、攻击、从高斯分布采样和交换来合成没有额外参数的风格。MaxStyle 和L2D 需要额外的网络模块和复杂的辅助任务来帮助生成新的风格。通常，AdvStyle 是与我们最相关的工作。因此，我们强调关键差异：1) AdvStyle 攻击图像上的风格，而我们使用渐进式攻击方法在多个特征空间上攻击风格；2) AdvStyle 使用相同的任务损失 (分割) 进行攻击和优化；相比之下，我们使用经典的分类损失来攻击风格，同时利用任务损失 (FSL) 来优化整个网络。

## StyleAdv: Meta Style Adversarial Training

**Task Formulation：**情节 $\mathcal{T}=((S,Q),Y)$ 随机采样为每一个元任务的输入，其中 $Y$ 表示情节图像相对于的 $\mathcal{C}^{tr}$ 的全局类别标签。通常，每个元任务被构建为一个 $N$ 类 $K$ 标记问题，也就是说，对于每一集 $T$，将具有 $K$ 个标记图像的 $N$ 个类采样为支持集 $S$，并使用另一个 $M$ 个图像的相同 $N$ 个类构成查询集 $Q$。FSL 或 CD-FSL 模型根据 $S$ 预测 $Q$ 中的图像属于 $N$ 个类别的概率 $P$。形式上，我们有 $|S| = NK$ ，$|Q| = NM$ ， $|P| = NM×N$。

### Overview of Meta Style Adversarial Learning

为了减轻视觉外观变化导致的性能下降，我们通过促进模型对识别各种风格的鲁棒性来解决 CD-FSL。因此，我们将我们的 FSL 模型暴露于源数据集中存在的图像风格之外的一些具有挑战性的虚拟风格。为此，我们提出了一种新颖的 StyleAdv 对抗训练方法。关键的是，我们不是对图像像素添加扰动，而是特别关注对抗性地扰动风格。我们的 StyleAdv 的总体框架如图 1 所示。

![image-20240324153338137](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403241643887.png)

我们的 StyleAdv 包含一个 CNN/ViT 主干 $E$，一个全局 FC 分类器 $f_{cls}$，和一个具有可学习参数$\theta_{E},\theta_{cls},\theta_{fsl}$ FSL 分类器 $f_{fsl}$。此外，我们还包括了我们的核心风格攻击方法、一种新颖的风格提取模块和 AdaIN。

总体而言，我们通过解决极小极大游戏来学习 StyleAdv。具体来说，极小极大游戏应该在每个元训练步骤中涉及两个迭代优化循环。特别是，

**内循环 (Inner loop)：**通过攻击原始源f风格来合成新的对抗性风格；生成的风格将增加当前网络的损失。

**外循环 (Outer loop)：**通过用原始风格和对抗性风格对源图像进行分类来优化整个网络；此过程将减少损失。

### Style Extraction from CNNs and ViTs

**Adaptive Instance Normalization (AdaIN)：**我们回顾了为 CNN 在风格迁移中提出的普通 AdaIN。特别是，AdaIN 表明实例级均值和标准差 (缩写为均值和标准差) 传达了输入图像的风格信息。将 $mean$ 和 $std$ 分别表示为 $\mu$ 和 $\sigma$，AdaIN (表示为 $A$) 表明 $F$ 的风格能通过将原始风格 $(\mu,\sigma)$ 替换为目标风格 $(\mu_{tgt},\sigma_{tgt})$ 迁移到 $F_{tgt}$ 的风格：

$\mathcal{A}(F,\mu_{tgt},\sigma_{tgt})=\sigma_{tgt}\frac{F-\mu(F)}{\sigma(F)}+\mu_{tgt}$

**Style Extraction for CNN Features：**如图 1 (b) 的上半部分所示，令 $F\in\mathcal{R}^{B\times C\times H\times W}$ 表示输入特征批次，其中 $B$，$C$，$H$，和 $W$ 分别表示特征 $F$ 的批量大小、通道、高度和宽度。与 AdaIN 一样，$F$ 的均值 $μ$ 和标准差 $\sigma$ 定义为：

$\mu(\mathrm{F})_{\mathrm{b,c}}=\frac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}F_{b,c,h,w}$

$\sigma(\mathrm F)_{\mathrm b,\mathrm c}=\sqrt{\dfrac{1}{HW}\sum_{h=1}^{H}\sum_{w=1}^{W}(F_{b,c,h,w}-\mu_{\mathrm b,\mathrm c}(F))^2+\epsilon}$

其中 $\mu,\sigma\in\mathcal{R}^{B\times C}$。

**Meta Information Extraction for ViT Features：**我们探索了将 ViT 特征的元信息提取为 CNN 的方式。直观地说，这种元信息可以看作是 ViT 的唯一“风格”。如图 1 (b) 所示，我们以一个输入批次的图像数据分割为 $P\times P$ 大小的块为例。ViT 编码器将一个批次的块编码为类别 token 和 一个块 token。为了与 AdaIN 比较，我们将 $F_0$ 改变为 $F\in{\mathcal R}^{B\times C\times P\times P}$ 形状。此时，我们可以计算块 tokens $F$ 的元信息，如式5和式6所示。本质上，请注意 transformer 将位置嵌入集成到块表示中，因此可以认为空间关系仍然存在于块 tokens 中。这支持了我们将块 tokens $F_0$ 转换为空间特征图 $F$。在某种程度上，这可以通过通过大小为 $P\times P$ 的内核对输入数据应用卷积来实现（如图 1 (b) 中的虚线箭头所示）。

### Inner Loop: Style Adversarial Attack Method

我们提出了一种新的风格对抗性攻击方法——快速风格梯度符号方法（Style-FGSM）来完成内循环。如图 1 所示，给定一个输入源集 $(\mathcal{T},Y)$，我们首先将其送到主干 $E$ 和 FC 分类器 $f_{cls}$ 产生全局分类损失 $L_{cls}$（如图 ① 路径所示）。在这个过程中，关键步骤是使风格的梯度可用。为此，设 $F_{\mathcal{T}}$ 表示 $\mathcal{T}$ 的特征，我们得到 $F_{\mathcal{T}}$ 的风格 $(\mu,\sigma)$。之后，我们将原始集的特征转变为 $\mathcal{A}(F_{\mathcal{T}},\mu,\sigma)$。转换后的特征用于实际上地前向传播。通过这种方式，我们将 $\mu$ 和 $\sigma$ 包含在了前向计算链中，因此，我们可以访问它们的梯度。

利用 ② 路径的梯度，我们和 FGSM 做的一样，分别为 $\mu$ 和 $\sigma$ 通过添加一个小的有符号梯度比率来攻击 $\mu$ 和 $\sigma$ 。

$\mu^{adv}=\mu+\epsilon\cdot\operatorname{sign}(\nabla_{\mu}J(\theta_{E},\theta_{f_{cls}},\mathcal{A}(F_{\mathcal{T}},\mu,\sigma),Y))$

$\sigma^{adv}=\sigma+\epsilon\cdot\mathrm{sign}(\nabla_{\sigma}J(\theta_{E},\theta_{f_{cls}},\mathcal{A}(F_{\mathcal{T}},\mu,\sigma),Y))$

其中 $J()$ 是分类预测和真值之间的交叉熵损失，即 $\mathcal{L}_{cls}$。受 PGD 随机开始的启发，我们还将随机噪声 $k_{RT}\cdot\mathcal{N}(0,I)$ 在攻击前添加到 $(\mu,\sigma)$ 。$\mathcal{N}(0,I)$ 指高斯噪声，$k_{RT}$ 是一个超参数。我们的 Style-FGSM 使我们能够生成“虚拟”和“硬”风格。

**Progressive Style Synthesizing Strategy：**为了防止高级对抗特征偏离，我们建议在渐进式策略中应用我们的 Style-FGSM。具体来说，嵌入模块 $E$ 有三个模块 $E_1$，$E_2$，$E_3$，对应特征$F_1$，$F_2$，$F_3$。对于第一个块，我们使用 $(\mu_{1},\sigma_{1})$ 来表示$F_1$ 的原始风格。对抗风格 $(\mu_1^{adv},\sigma_1^{adv})$ 直接通过公式 7 和公式 8 得到。对于后续块，当前块 $i$ 的攻击信号是从块 $1$ 累计到块 $i-1$ 的。以第二个块为例，块特征 $F_2$ 不是简单地由 $E_{2}(F_{1})$ 提取。相反，我们有 $F_{2}^{'}=E_{2}(F_{1}^{adv})$，其中 $F_{1}^{adv}=\mathcal{A}(F_{1},\mu_{1}^{adv},\sigma_{1}^{adv})$。在 $F_{2}^{'}$ 上的攻击产生对抗风格 $(\mu_{2}^{adv},\sigma_{2}^{adv})$。因此，我们为最后一个块生成了 $(\mu_{3}^{adv},\sigma_{3}^{adv})$。渐进式攻击策略的说明附在附录中。

**Changing Style Perturbation Ratios：**与普通 FGSM 或 PGD 不同，我们的风格攻击算法有望合成具有多样性的新风格。因此，我们没有使用固定的攻击比率 $\epsilon$ ，而是从候选列表 $\epsilon_{list}$ 中随机抽取 $\epsilon$ 作为当前攻击比率。尽管 $\epsilon$ 的随机性，但我们仍然以更具挑战性的方向合成风格， $\epsilon$ 仅影响程度。

### Outer Loop: Optimize the StyleAdv Network

对于每个以干净的情节 $\mathcal{T}$ 作为输入的元训练迭代，我们的内循环生成对抗性风格 $(\mu_1^{adv},\sigma_1^{adv})$ ，$(\mu_{2}^{adv},\sigma_{2}^{adv})$，$(\mu_{3}^{adv},\sigma_{3}^{adv})$。如图 1 所示，外循环的目标是使用干净的特征 $F$ 和用作训练数据的风格攻击特征 $F^{adv}$ 来优化整个 StyleAdv。通常，干净的情节特征 $F$ 可以直接作为 $E(T)$ 获得如路径 ③ 所示。

在路径 ④ 中，我们通过将 $F$ 的原始风格迁移到相应的对抗性攻击风格来获得 $F^{adv}$。与渐进式 Style-FGSM 类似，$F_{1}^{adv}~=~\mathcal{A}(E_{1}(\mathcal{T}),\mu_{1}^{adv},\sigma_{1}^{adv})$，$F_{2}^{adv}~=~\mathcal{A}(E_{2}(\mathcal{T}),\mu_{2}^{adv},\sigma_{2}^{adv})$，$F_{3}^{adv}~=~\mathcal{A}(E_{3}(\mathcal{T}),\mu_{3}^{adv},\sigma_{3}^{adv})$。最后，通过将平均池化层应用于 $F_{3}^{adv}$ 来获得 $F^{adv}$。设置跳过概率 $p_{skip}$ 来决定是否跳过当前攻击。对干净的特征 $F$ 和风格攻击特征 $F^{adv}$ 进行 FSL 任务会产生两个 FSL 预测 $P_{fsl}$，$P_{fsl}^{adv}$ 和两个 FSL 分类损失 $\mathcal{L}_{fsl}$，$\mathcal{L}_{fsl}^{adv}$。

此外，尽管 $F^{adv}$ 的风格是从 $F$ 转变而来，我们还是建议语义内容应与 wave-SAN 保持一致。因此，我们为预测 $P_{fsl}$ 和 $P_{fsl}^{adv}$ 添加了一致性约束，从而产生了一致性损失 $\mathcal{L}_{cons}$：

$\mathcal{L}_{cons}=\mathrm{KL}(P_{fsl},P_{fsl}^{adv})$

其中 KL 为 Kullback-Leibler 散度损失。此外，我们有全局分类损失 $\mathcal{L}_{cls}$。这保证了 $\theta_{cls}$ 被优化为 style-FGSM 提供正确的梯度。StyleAdv 的最终原目标为：

$\mathcal{L}=\mathcal{L}_{fsl}+\mathcal{L}_{fsl}^{adv}+\mathcal{L}_{cons}+\mathcal{L}_{cls}.$

注意，我们的 StyleAdv 与模型无关，并且与现有的 FSL 和 CD-FSL 方法正交。

### Network Inference 

**Applying StyleAdv Directly for Inference：**我们的 StyleAdv 有助于使 CD-FSL 模型对风格转换更加稳健。一旦模型经过元训练，我们就可以通过将测试集输入 $E$ 和 $f_{cls}$ 单元来直接使用它进行推理。概率最高的类别将被作为预测结果。

**Finetuning StyleAdv Using Target Examples：**如之前的研究所示，在目标样例上对 CD-FSL 模型进行微调有助于提高模型的性能。因此，为了进一步提升 StyleAdv 的性能，我们还为其配备了形成升级版本的 fintuning 策略("StyleAdv-FT")。具体来说，就像在 ATA-FT 中一样，对于每个新的测试集，我们增加新的支持集，形成伪集作为训练数据，用于调整元训练模型。

## Experiments

**Datasets：**我们采用 BSCD-FSL 和 FWT 中提出的两个 CD-FSL 基准。两者都以 miniImagenet 作为源数据集。两个不相交的集合从 mini-Imagenet 中分离出来，形成 $\mathcal{D}^{tr}$ 和 $\mathcal{D}^{eval}$。本文将ChestX、ISIC、EuroSAT、CropDisease、CUB、Cars、Places、Plantae等 8 个数据集作为新的目标数据集。BSCD-FSL 基准中包括的前四个数据集涵盖从 x 射线到皮肤镜下皮肤病变的医学图像，以及从卫星图片到植物病害照片的自然图像。而后四个数据集关注更细粒度的概念，如鸟类和汽车，则包含在 FWT 中。这 8 个目标数据集分别作为测试集 $\mathcal{D}^{te}$。

**Network Modules：**对于典型的基于 CNN 的网络，沿用以往的 CD-FSL 方法，选择 ResNet10 作为嵌入模块，选择 GNN 作为 FSL 分类器；对于新兴的基于 ViT 的网络，继 PMF 之后，我们分别使用 ViT-small 和 ProtoNet 作为嵌入模块和 FSL 分类器。请注意，ViT-small 是在 ImageNet1K 上通过 DINO 进行预训练的，就像在 PMF 中一样。$f_{cls}$ 由一个完全连接的层构建。

**Implementation Details：**进行 5 类 1 样本和 5 类 5 样本设置。以 ResNet10 为骨干，对网络进行 200 个 epoch的元训练，每个 epoch 包含 120 个元任务。使用学习率为 0.001 的 Adam 作为优化器。以 ViT-small 为骨干，元训练阶段为 20 个 epoch，每个 epoch 包含 2000 个元任务。初始学习率为 5e-5 和 0.001 的 SGD 分别用于优化 $E()$ 和 $f_{cls}$。Style-FGSM 攻击器的 $\epsilon_{list}$，$k_{RT}$ 设为 $[0.8,0.08,0.008]$，$\frac{16}{255}$。随机跳过攻击的概率 $p_{skip}$ 从 ${0.2, 0.4}$ 中选择。我们用 1000 个随机采样的集评估我们的网络，并报告 95% 置信区间的平均准确率 (%)。我们的 “StyleAdv” 和 "StyleAdv-FT" 的结果都被报告了。微调的细节见附录。基于 ResNet-10 的模型在单个 GeForce GTX 1080 上进行训练和测试，而基于 ViT-small 的模型需要单个 NVIDIA GeForce RTX 3090。

![image-20240325112744723](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403251127966.png)

### Comparison with the SOTAs

我们将 StyleAdv/StyleAdv-FT 与几种最具代表性和竞争力的 CD-FSL 方法进行了比较。具体而言，以 ResNet-10 (缩写为 RN10) 为骨干，介绍了 GNN、FWT、LRP、ATA、AFA、wave-SAN、Finetune、NSAE、BSR 等 9 种方法作为我们的竞争对手。其中，前六个竞争者是直接用于推理的基于元学习的方法，因此我们将我们的 “StyleAdv” 与它们进行比较，以进行公平的比较。通常，GNN 作为基础模型。Fine-tune、NSAE、BSR和 ATA-FT (由微调 ATA 形成) 在推理过程中都需要微调模型，因此使用我们的 “StyleAdv-FT”。以 ViT 为骨干，比较了最新和最具竞争力的PMF (FSL 的 SOTA 方法)。为了公平比较，我们遵循 PMF 中提出的相同流程。注意，我们只使用一个单一的源域来推广 CD-FSL 模型。不考虑那些使用额外训练数据集的方法，如 STARTUP、meta-FDMixup 和 DSL。对比结果如表 1 所示。

对于所有结果，我们的方法明显优于所有列出的 CD-FSL 竞争对手，并达到了最新的性能。我们的 StyleAdv-FT (ViT-small) 在 5 类 1 样本和 5 类 5 样本上的平均成绩分别为 58.57% 和 74.06%。我们的 StyleAdv (RN10) 和StyleAdv-FT (RN10) 也胜过所有基于元学习或基于迁移学习 (微调) 的方法。除了最先进的精度，我们还有其他值得一提的观察结果。1) 我们证明了我们的 StyleAdv 方法对于基于 CNN 的模型和基于 ViT 的模型都是通用的解决方案。通常，基于 ResNet10，我们的 StyleAdv 和 StyleAdv-FT 在 5 样本设置下将基本 GNN 提高了 4.93% 和 8.44%。基于 ViT-Small，在大多数情况下，我们的 StyleAdv-FT 明显优于 PMF。在其他 FSL 或 CD-FSL 方法上构建StyleAdv 的更多结果可以在附录中找到。2) 比较 FWT、LRP、ATA、AFA、waveSAN 和我们的 StyleAdv，我们发现 StyleAdv 表现最好，其次是 wave-SAN，然后是 AFA、ATA、FWT 和 LRP。这一现象表明，通过解决视觉移位问题来解决 CD-FSL 问题确实比其他方法更有效，例如通过扰动图像特征 (AFA) 或图像像素 (ATA) 进行对抗性训练，在 FWT 中变换归一化层，在 LRP 中进行解释指导训练。3)对于 StyleAdv 和 wave-SAN 之间的比较，我们注意到 StyleAdv 在大多数情况下都优于 wave-SAN。这表明我们的 StyleAdv 生成的风格比 wave-SAN 中提出的风格增强方法更有利于学习稳健的 CD-FSL 模型。这证明了我们合成更具挑战性 (“硬和虚拟”) 风格的想法是正确的。4) 总体而言，大规模预训练模型对 CD-FSL 有明显的促进作用。以 1 样本设置为例，StyleAdv-FT (ViT-small)平均提高StyleAdv-FT (RN10) 9.16%。然而，我们表明，性能改进在不同的目标域上差异很大。一般来说，对于领域差异相对较小的目标数据集，如 CUB 和 Plantae，模型受益较多；否则，改善是有限的。5)我们还发现，在跨域场景下，目标域的微调模型 (如 NSAE、BSR) 确实比纯粹基于元学习的方法 (如 FWT、LRP 和 wave-SAN) 更具优势。然而，使用极少量的例子来微调模型，例如，5 类 1 样本的设置比使用相对较多的样本设置要困难得多。这也许可以解释为什么那些基于微调的方法不进行单次设置的实验。

**Effectiveness of Style-FGSM Attacker：**为了展示渐进式风格合成策略的优势，以及通过改变扰动比进行攻击，我们将 Style-FGSM 与几种变体进行了比较，并在图 2 中报告了结果。具体来说，对于图2 (a)，我们将 Style-FGSM 与单独攻击块的变体进行比较。结果表明，在大多数情况下，渐进式攻击优于朴素个体策略。对于图2 (b)，为了演示固定攻击比率对性能的影响，我们还使用不同的 $\epsilon_{list}$ 进行了实验。由于我们将其设为 $[0.8,0.08,0.008]$，因此我们选择了 $[0.8]$，$[0.08]$ 和 $[0.008]$ 三个不同的选项。从结果中，我们首先注意到单一固定比率可以达到最佳结果。然而，在大多数情况下，从候选池中采样攻击率可以获得最佳结果。

![image-20240325114538192](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403251145588.png)



### More Analysis

**Visualization of Hard Style Generatio：**为了帮助直观地理解我们方法的“硬”样式生成，如图 3 所示，我们将 StyleAdv 与 wave-SAN 进行了几个可视化比较。1) 如图 3 (a) 所示，我们展示了由 wave-SAN 和 StyleAdv 生成的风格化图像。通过对输入图像应用样式增强方法实现可视化。具体来说，对于 wave-SAN，样式与另一个随机采样的源图像交换；对于 StyleAdv，给出了攻击 style 的结果，其价值为：我们观察到 wave-SAN 倾向于随机交换全局视觉外观，例如输入图像的颜色。相比之下，StyleAdv 更喜欢干扰对识别图像类别至关重要的区域。例如，猫的皮毛和狗的关键部位(脸和脚)。这些观察结果直观地支持了我们的说法，即我们的 StyleAdv 合成了比 wave-SAN 更硬的样式。2)为了定量评估 StyleAdv 是否在训练阶段引入了更具挑战性的风格，如图 3 (b) 所示，我们将元训练损失可视化。结果表明，wave-SAN 的扰动损耗在原始损耗附近振荡，而 StyleAdv 明显增加了原始损耗。这些现象进一步验证了我们将数据向更困难的方向扰动，从而在很大程度上推动了样式生成的极限。3) 为了进一步显示 StyleAdv 相对于 wave-SAN 的优势，如图 3 (c) 所示，我们将元训练后的 wave-SAN 和 StyleAdv 提取的高级特征可视化。选择 mini-Imagenet 的 5 个类 (用不同颜色表示)。T-SNE 用于降低特征维数。结果表明，StyleAdv 扩大了类间距离，使类更容易区分。



![image-20240325114737800](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403251147175.png)

**Why Attack Styles Instead of Images or Features?**一个自然的问题可能是，为什么我们选择攻击风格而不是其他目标，例如 AQ， MDAT 和 ATA 中的输入图像或 Shen 等人和 AFA 中的特征？为了回答这个问题，通过修改我们方法的攻击目标，我们比较了 StyleAdv 攻击风格和攻击图像和特征。

5 类 1 样本和 5 类 5 样本结果见表 2。我们强调几点。1) 我们注意到攻击图像、特征和风格都改进了基本 GNN 模型 (如表 1 所示)，这表明它们都通过对抗性攻击提高了模型的泛化能力。有趣的是，我们的“攻击图像”/“攻击特征”的结果甚至优于设计良好的 CD-FSL 方法 ATA 和 AFA (见表 1)；2) 与攻击图像和特征相比，我们的方法有明显的优势。这再次表明了处理视觉样式以缩小 CD-FSL 的领域差距问题的优越性。

![image-20240325115446299](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main/202403251154517.png)

**Is Style-FGSM Better than Other Style Augmentation Methods?**为了展示我们的 Style-FGSM 相对于其他风格增强方法的优势，我们引入了几个竞争对手，包括 “StyleGaus”、MixStyle、AdvStyle 和 DSU。通常，“StyleGaus” 将随机高斯噪声添加到风格中，作为简单但合理的基线引入。MixStyle，AdvStyle 和 DSU 最初是为其他任务设计的，例如分割和领域生成。结果如表 3 所示。将 StyleGuas 的结果与表 1 中报告的结果进行比较，我们发现通过简单地添加随机噪声在特征级别上扰动风格也提高了基本 GNN，甚至在一些目标数据集上超过了一些 CD-FSL 竞争对手。这种现象与增加样式分布有助于提高 CD-FSL 方法的洞察力是一致的。至于我们的 Style-FGSM 和其他高级风格增强竞争对手的比较，我们发现 Style-FGSM 在5 类 1 样本和 5 类 5 样本设置上都比所有 MixStyle, AdvStyle 和 DSU 表现得更好。通常，MixStyle 和 DSU 都生成虚拟样式，但是它们的新样式仍然相对容易。这说明我们的硬风格在更大程度上提升了模型。AdvStyle 生成虚拟和硬 (对抗) 风格。但是，它还是不如我们。这表明我们的方法在潜在特征空间中进行攻击，并采用两个独立的任务进行攻击和优化。

## Conclusion

本文提出了一种新的 CD-FSL 模型不可知的方法。至关重要的是，为了缩小通常以视觉变化形式出现的领域差距，StyleAdv 解决了风格对抗学习的最小化游戏：首先向源风格添加扰动，增加当前模型的损失，然后通过强制其识别干净和风格扰动数据来优化模型。此外，我们还提出了一种新的渐进式对抗性攻击方法，称为 Style-FGSM。Style-FGSM 综合了各种“硬”和“虚拟”的风格，通过添加签名渐变到原来的干净的风格。这些生成的风格支持 StyleAdv 的最大步长。直观地，通过将 CD-FSL 暴露于比源数据集中存在的有限真实风格更具挑战性的对抗风格中，提高了模型的泛化能力。我们的 StyleAdv 改进了基于 CNN 和基于 ViT 的模型。大量的实验表明，我们的 StyleAdv 构建了新的SOTA。