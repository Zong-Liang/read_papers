# VM-UNet: Vision Mamba UNet for Medical Image Segmentation

## 针对任务

医学图像分割。

## 动机

基于CNN的模型具有长程建模局限，基于transformer的模型性能受到二次计算复杂度$O(n^2)$的阻碍，提出了纯基于状态空间模型的Vision Mamba UNet (VM-UNet)，同时具有长程建模能力和线性计算复杂度，并在Synapse、ISIC17、ISIC18三个数据集上验证了VM-UNet在医学图像分割任务上的性能。

## Preliminaries

在现代基于 SSM 的模型中，即结构化状态空间模型 (S4) 和 Mamba，都依赖于经典的连续系统，该系统通过中间隐式状态 $h(t)∈R^N$映射到输出$y(t) ∈ R$，表示为$x(t) ∈ R$。上述过程可以表示为线性常微分方程 (ODE)：

$h'(t)=Ah(t)+Bx(t)$

$y(t)=Ch(t)$

其中，$A∈R^{N×N}$ 代表状态矩阵，$B∈R^{N×1}$ 和 $C∈R^{N×1}$ 表示投影参数。

S4 和 Mamba 离散化这个连续系统以使其更适合深度学习场景。具体来说，他们引入了时间尺度参数$\Delta $，并使用固定的离散化规则将 $A$ 和 $B$ 转换为连参数 $\overline{\mathrm{A}}$和 $\overline{\mathrm{B}}$。通常，零阶保持 (ZOH) 被用作离散化规则，可以定义为：

$\begin{aligned}\overline{\mathbf{A}}&=\exp(\boldsymbol{\Delta}\mathbf{A})\end{aligned}$

$\overline{\mathbf{B}}=(\boldsymbol{\Delta}\mathbf{A})^{-1}(\exp(\boldsymbol{\Delta}\mathbf{A})-\mathbf{I})\cdot\boldsymbol{\Delta}\mathbf{B}$

离散化之后，基于 SSM 的模型可以通过两种方式计算：线性递归或全局卷积，分别定义如下：

$h'(t)=\overline{\mathbf{A}}h(t)+\overline{\mathbf{B}}x(t)$

$y(t)=\mathbf{C}h(t)$

$\overline{K}=(\mathrm{C}\overline{\mathrm{B}},\mathrm{C}\overline{\mathrm{A}}\overline{\mathrm{B}},\ldots,\mathrm{C}\overline{\mathrm{A}}^{L-1}\overline{\mathrm{B}})$

$y=x*\overline{\mathrm{K}}$

其中，$\overline{\mathrm{K}}∈R^L$ 代表结构化卷积核，$L$ 表示输入序列 $x$ 的长度。

## 结构图

![](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202402261111160.png)

VM-UNet由三个主要部分组成：编码器、解码器和跳跃连接。编码器由VMamba的VSS块组成进行特征提取，以及用于下采样的补丁合并操作。相反，解码器包括VSS块和补丁扩展操作来恢复分割结果的大小。对于跳跃连接组件，为了突出原始纯基于 SSM 的模型的分割性能，本文采用了最简单的加法运算形式。

## 方法细节

### Vision Mamba UNet (VM-UNet)

VM-UNet 包括一个补丁嵌入层、编码器、解码器、最终投影层和跳跃连接。

补丁嵌入层将输入图像$x∈R^{H×W ×3}$划分为大小为$4 × 4$的非重叠块，然后将图像的维度映射到$C$，$C$默认为 $96$。这个过程导致嵌入图像$x′∈R^{H/4×W/4×C}$。最后，使用层归一化对$x'$进行归一化，然后将其输入到编码器中进行特征提取。编码器由四个阶段组成，在前三个阶段的末尾应用补丁合并操作，以减少输入特征的高度和宽度，同时增加通道数。在四个阶段使用$[2,2,2,2]$VSS块，每个阶段的通道计数为$[C, 2C, 4C, 8C]$。

同样，解码器分为四个阶段。在最后三个阶段的开始，利用补丁扩展操作减少特征通道的数量，增加高度和宽度。在四个阶段，我们利用$[2,2,2,1]$VSS 块，每个阶段的通道数为$[8C, 4C, 2C, C]$。在解码器之后，使用最终投影层来恢复特征的大小以匹配分割目标。具体来说，通过补丁扩展进行$4$次上采样以恢复特征的高度和宽度，然后使用投影层来恢复通道数。

对于跳跃连接，采用了一个简单的加法操作，因此不会引入任何额外的参数。

### VSS block

VSS 块是 VM-UNet 的核心模块。在进行层归一化后，输入被分成两个分支。在第一个分支中，输入通过一个线性层和一个激活函数。在第二个分支中，输入通过线性层、深度可分离卷积和激活函数进行处理，然后将其输入到 2D-Selective-Scan (SS2D) 模块中进行进一步的特征提取。随后，使用层归一化对特征进行归一化，然后使用第一个分支的输出执行逐元素相乘以合并两条路径。最后，使用线性层对特征进行混合，并将结果与残差连接相结合，形成VSS块的输出。在本文中，SiLU 默认用作激活函数。

> SiLU，即 Sigmoid Linear Unit，也被称为 Swish 激活函数。它是由Google研究员于2017年提出的一种激活函数。SiLU 函数的定义如下：$\mathrm{SiLU}(x)=x\cdot\sigma(x)$，其中 $\sigma(x)$ 是 Sigmoid 函数：$\sigma(x)=\frac1{1+e^{-x}}$，SiLU 函数的图像类似于 ReLU，但具有更平滑的形状。与 ReLU 相比，SiLU 函数在某些情况下可以提供更好的性能，特别是在深度神经网络中。SiLU 激活函数在保留非线性特性的同时，还具有很好的平滑性和渐变消失问题的缓解。

![image-20240226123037294](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202402261230434.png)

SS2D 由三个部分组成：扫描扩展操作、S6 块和扫描合并操作。扫描扩展操作沿着四个不同的方向展开输入图像 (左上角到右下角，右下角到左上角，右上角到左下角，左下角到右上角) 成序列。然后，这些序列由 S6 块处理进行特征提取，确保彻底扫描来自各个方向的信息，从而捕获不同的特征。随后，扫描合并操作和合并这四个方向的序列，将输出图像恢复到与输入相同的大小。从 Mamba 导出的 S6 块通过基于输入调整 SSM 的参数，在 S4 之上引入了选择性机制。这使得模型能够区分和保留相关信息，同时过滤掉不相关的信息。S6块的伪代码如算法 1 所示。

![image-20240226123552191](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202402261235300.png)

### Loss function

用最基本的二元交叉熵和骰子损失 (BceDice 损失) 和交叉熵和骰子损失 (CeDice 损失) 分别作为二元和多类分割任务的损失函数。

<img src="https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202402261238867.png" alt="image-20240226123855764" style="zoom:50%;" />

其中，$N$ 表示样本总数，$C$ 表示类别的总数。${y}_i$、$\hat{y}_i$ 分别表示真实标签和预测标签。${y}_{i,c}$ 是一个指标，如果样本$i$属于类别 $c$，则等于1，否则为0。$\hat{y}_{i,c}$ 是模型预测样本$i$属于类别$c$的概率。$|X|$、$|Y|$ 分别代表真值和预测值，$\lambda_1$、$\lambda_2$ 指的是损失函数的权重，默认情况都设置为1。

## Experiments

### Datasets

ISIC17 and ISIC18 datasets: 以 7:3 的比例拆分数据集，用作训练集和测试集。对于 ISIC17 数据集，训练集由 1,500 张图像组成，测试集包含 650 张图像。对于 ISIC18 数据集，训练集包含 1,886 张图像，而测试集包含 808 张图像。对于这两个数据集，我们对平均交集 over Union (mIoU)、Dice Similarity Coefficient (DSC)、Accuracy (Acc)、Sensitivity (Sen) 和特异性 (Spe) 这几个指标进行了详细的评估。

Synapse multi-organ segmentation dataset (Synapse): 18 个案例用于训练，12 个案例用于测试。对于这个数据集，我们用 Dice Similarity Coefficient (DSC) 和 95% Hausdorff Distance (HD95) 作为评估指标。

### Implementation details

将 ISIC17 和 ISIC18 数据集中的图像调整为 256×256，将 Synapse 数据集中的图像调整为 224×224。为了防止过拟合，采用了数据增强技术，包括随机翻转和随机旋转。BceDice 损失函数用于 ISIC17 和 ISIC18 数据集，而 Synapse 数据集采用 CeDice 损失函数。我们将批量大小设置为 32，并使用初始学习率为 1e-3 的 AdamW 优化器。 CosineAnnealingLR 被用作调度程序，最多 50 次迭代，最小学习率为 1e-5。训练时期设置为 300。对于 VM-UNet，我们使用在 ImageNet-1k 上预训练的 VMamba-S 初始化编码器和解码器的权重。所有实验均在单个 NVIDIA RTX A6000 GPU 上进行。

### Main results

![image-20240226125448760](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202402261254870.png)

![image-20240226125516322](https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202402261255433.png)

对于 ISIC17 和 ISIC18 数据集，VM-UNet 在 mIoU、DSC 和 Acc 指标方面优于其他模型。对于 Synapse 数据集，VM-UNet 也取得了具有竞争力的性能。例如，我们的模型在 DSC 和 HD95 指标中超过了第一个纯基于 Transformer 的模型 Swin-UNet。结果表明基于SSM的模型在医学图像分割任务中的优越性。

<img src="https://cdn.jsdelivr.net/gh/ZL85/ImageBed@main//202402261255878.png" alt="image-20240226125556733" style="zoom:50%;" />

表明更有效的预训练权重显著提高了VM-UNet的下游性能，表明VM-UNet受预训练权重的影响很大。

## Conclusions and Future works

### Conclusions

在本文中，我们首次引入了纯基于 SSM 的医学图像分割模型，将 VM-UNet 作为基线。为了利用基于SSM的模型的能力，我们使用VSS块构建VM-UNet，并使用预训练的VMamba-S初始化其权重。在皮肤病变和多器官分割数据集上进行了综合实验，表明纯基于SSM的模型在医学图像分割任务中具有很强的竞争力，未来值得深入探索。

### Future works

- 基于SSMs的机制，更适合分割任务的设计模块。
- VM-UNet 的参数计数约为 30M，为通过手动设计或其他压缩策略简化 SSM 提供了机会，从而有助于它们在现实世界医疗场景中的适用性。
- 鉴于 SSM 在捕获长序列中的信息方面的优势，进一步研究更高分辨率的分割性能将是有价值的。
- 探索SSM在其他医学成像任务中的应用，如检测、配准和重建等。

## vocabulary

|                单词                |       音标        | 词性  |           中文           |
| :--------------------------------: | :---------------: | :---: | :----------------------: |
|               realm                |      /relm/       |  n.   |           领域           |
|          in the realm of           |                   |       |        在…领域里         |
|            extensively             |   ɪkˈstensɪvli/   | adv.  |                          |
|              explore               |   /ɪkˈsplɔː(r)/   |  v.   |           探索           |
|              exhibit               |    /ɪɡˈzɪbɪt/     |  v.   |          表现出          |
|       exhibit limitations in       |                   |       |  在...方面表现出局限性   |
|  long-range modeling capabilities  |                   |       |       长程建模能力       |
|               hamper               |    /ˈhæmpə(r)/    |  v.   |           阻碍           |
|              whereas               |    /ˌweərˈæz/     | conj. |           然而           |
|             quadratic              |   /kwɒˈdrætɪk/    | adj.  |          二次的          |
| quadratic computational complexity |                   |       |  二次计算复杂度$O(n^2)$  |
|     State Space Models (SSMs)      |                   |       |       空间状态模型       |
|             exemplify              |   ɪɡˈzemplɪfaɪ/   |  v.   |        是…的典型         |
|               emerge               |     /ɪˈmɜːdʒ/     |  v.   |           出现           |
|             promising              |    /ˈprɒmɪsɪŋ/    | adj.  |         有前途的         |
|              approach              |    /əˈprəʊtʃ/     |  n.   |           方法           |
|               excel                |     /ɪkˈsel/      |  v.   |        精通，擅长        |
|  modeling long-range interactions  |                   |       |       建模长程交互       |
|              maintain              |    /meɪnˈteɪn/    |  v.   |           保持           |
|  linear computational complexity   |                   |       |      线性计算复杂度      |
|              leverage              |   /ˈliːvərɪdʒ/    |  v.   |           利用           |
|              capture               |   /ˈkæptʃə(r)/    |  v.   |           捕获           |
|             extensive              |   /ɪkˈstensɪv/    | adj.  |      广泛的、大量的      |
|       contextual information       |                   |       |        上下文信息        |
|            asymmetrical            |  /ˌeɪsɪˈmetrɪk/   | adj.  |         非对称的         |
|              conduct               |    /kənˈdʌkt/     |  v.   |        实施、进行        |
|           comprehensive            | /ˌkɒmprɪˈhensɪv/  | adj.  |     综合性的、全面的     |
|              indicate              |    /ˈɪndɪkeɪt/    |  v.   |           表明           |
|       performs competitively       |                   |       |      表现具有竞争力      |
|       to our best knowledge        |                   |       |        据我们所知        |
|             automated              |                   |       |         自动化的         |
|             physicians             |     /fɪˈzɪʃn/     |  n.   |         内科医生         |
|            pathological            |  /ˌpæθəˈlɒdʒɪkl/  | adj.  |         病理学的         |
|             diagnosis              |  /ˌdaɪəɡˈnəʊsɪs/  |  n.   |           诊断           |
|              thereby               |    /ˌðeəˈbaɪ/     | adv.  |           因此           |
|         the efficiency of          |                   |       |        ...的效率         |
|            patient care            |                   |       |         患者护理         |
|            demonstrate             |  /ˈdemənstreɪt/   |  v.   |           展示           |
|             remarkable             |   /rɪˈmɑːkəbl/    | adj.  |    引人注目的，非凡的    |
|              various               |    /ˈveəriəs/     | adj.  |        各种各样的        |
|            particularly            |  /pəˈtɪkjələli/   | adv.  |           尤其           |
|       as a representative of       |                   |       |       作为...代表        |
|            scalability             | /ˌskeɪləˈbɪləti/  |  n.   |         可扩展性         |
|             subsequent             |  /ˈsʌbsɪkwəntli/  | adj.  |          之后的          |
|              pioneer               |  /ˌpaɪəˈnɪə(r)/   |  n.   |       拓荒者、先锋       |
|               employ               |     /ɪmˈplɔɪ/     |  v.   |        使用、利用        |
|               phase                |      /feɪz/       |  n.   |           阶段           |
|              utilize               |   /ˈjuːtəlaɪz/    |  v.   |        利用、使用        |
|            significant             |  /sɪɡˈnɪfɪkənt/   | adj.  |          显著的          |
|            acquisition             |   /ˌækwɪˈzɪʃn/    |  n.   |           获取           |
|            incorporate             |  /ɪnˈkɔːpəreɪt/   |  v.   |           包含           |
|              parallel              |    /ˈpærəlel/     |  n.   |          平行的          |
|           simultaneously           | /ˌsɪmlˈteɪniəsli/ | adv.  |          同时地          |
|            furthermore             |  /ˌfɜːðəˈmɔː(r)/  | adv.  |        此外、而且        |
|            nevertheless            |   /ˌnevəðəˈles/   | adv.  |        然而、不过        |
|              inherent              |    /ɪnˈherənt/    | adj.  |      内在的、固有的      |
|         be constrained by          |   /kənˈstreɪnd/   | adj.  |        受...约束         |
|            considerably            |  /kənˈsɪdərəbli/  | adv.  |      非常、相当多地      |
|               hinder               |    /ˈhɪndə(r)/    |  v.   |        阻碍、妨碍        |
|             inadequate             |   /ɪnˈædɪkwət/    | adj.  |     不充分的、不足的     |
|             suboptimal             |  /ˌsʌbˈɒptɪməl/   | adj.  | 次最优的、未达最佳标准的 |
|              superior              |  /suːˈpɪəriə(r)/  | adj.  |          更强的          |
|            in terms of             |                   |       |        在...方面         |
|         dense predictions          |                   |       |         密集预测         |
|            shortcomings            |   /ˈʃɔːtkʌmɪŋ/    |  n.   |           缺点           |
|               compel               |     /kəmˈpel/     |  v.   |        强迫、迫使        |
|          compel sb. to do          |                   |       |       强迫某人去做       |
|               novel                |      /ˈnɒvl/      |  n.   |          新颖的          |
|              capable               |    /ˈkeɪpəbl/     | adj.  |   有能力的、可以...的    |
|              attract               |     /əˈtrækt/     |  v.   |       引起…的兴趣        |
|            considerable            |  /kənˈsɪdərəbl/   | adj.  |    极大的、相当重要的    |
|             classical              |    /ˈklæsɪkl/     | adj.  |          经典的          |
|             establish              |    /ɪˈstæblɪʃ/    |  v.   |           建立           |
|              exhibit               |    /ɪɡˈzɪbɪt/     |  v.   |          表现出          |
|          with respect to           |                   |       |           关于           |
|            additionally            |   /əˈdɪʃənəli/    | adv.  |        另外、此外        |
|            substantial             |   /səbˈstænʃl/    | adj.  |          大量的          |
|         across many fields         |                   |       |        在许多领域        |
|           aforementioned           |  /əˈfɔːmenʃənd/   | adj.  |    上述的、前面提及的    |
|              showcase              |    /ˈʃəʊkeɪs/     |  v.   |        展示、展现        |
|             potential              |    /pəˈtenʃl/     |  n.   |           潜力           |
|              composed              |    /kəmˈpəʊzd/    | adj.  |        由……组成的        |
|           be composed of           |                   |       |        由...组成         |
|         feature extraction         |                   |       |         特征提取         |
|             conversely             |   /ˈkɒnvɜːsli/    | adv.  |     相反地、反过来说     |
|              comprise              |    /kəmˈpraɪz/    |  v.   |           包含           |
|              restore               |   /rɪˈstɔː(r)/    |  v.   |           恢复           |
|             highlight              |    /ˈhaɪlaɪt/     |  v.   |        突出、强调        |
|               adopt                |     /əˈdɒpt/      |  v.   |           采用           |
|         additive operation         |                   |       |         加法运算         |
|                i.e.                |     /ˌaɪ ˈiː/     | abbr. |            即            |
|             elaborate              |    /ɪˈlæbərət/    |  v.   |         详尽阐述         |
|            elaborate on            |                   |       |         详细说明         |

