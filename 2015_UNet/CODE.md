# UNet

```python
import torch.nn as nn
import torch


# 允许用户根据需要获取不同类型的激活函数。如果提供的 activation_type 不是 PyTorch 中的标准激活函数名称，那么它将默认返回 ReLU 激活函数。
def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


# 简化创建包含多个卷积层的网络块的过程，只需指定输入通道数、输出通道数和卷积层的数量，即可自动生成对应的网络块。
def _make_nConv(in_channels, out_channels, nb_Conv, activation="ReLU"):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


# 实现了一个卷积层后接批归一化和激活函数的整体操作，用于构建深度神经网络中的各个层。
class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation="ReLU"):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


# 实现了一个下采样模块，用于在深度神经网络中降低特征图的空间尺寸。
class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation="ReLU"):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


# 实现了一个上采样模块，用于在深度神经网络中增加特征图的空间尺寸。
class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation="ReLU"):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), 2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        out = self.up(x)
        x = torch.cat([out, skip_x], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)


# 定义了一个基础的 UNet 模型，它是一个用于图像分割的卷积神经网络。UNet 模型结构由编码器（下采样路径）和解码器（上采样路径）组成，其中编码器负责提取图像特征，解码器则负责将提取的特征进行上采样和融合，最终输出分割结果。
class UNet_base(nn.Module):
    def __init__(self, n_channels=3, n_classes=9):
        """
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        if n_classes != 1:
            self.n_classes = n_classes + 1
        # Question here
        in_channels = 64
        self.inc = ConvBatchNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)
        self.down4 = DownBlock(in_channels * 8, in_channels * 8, nb_Conv=2)
        self.up4 = UpBlock(in_channels * 16, in_channels * 4, nb_Conv=2)
        self.up3 = UpBlock(in_channels * 8, in_channels * 2, nb_Conv=2)
        self.up2 = UpBlock(in_channels * 4, in_channels, nb_Conv=2)
        self.up1 = UpBlock(in_channels * 2, in_channels, nb_Conv=2)
        self.outc = nn.Conv2d(in_channels, self.n_classes, kernel_size=(1, 1))
        if n_classes == 1:
            self.last_activation = nn.Sigmoid()
        else:
            self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        # print(x.shape)  # [4, 3, 224, 224]
        x1 = self.inc(x)
        # print(x1.shape)  # [4, 64, 224, 224]
        x2 = self.down1(x1)
        # print(x2.shape)  # [4, 128, 224, 224]
        x3 = self.down2(x2)
        # print(x3.shape)  # [4, 256, 224, 224]
        x4 = self.down3(x3)
        # print(x4.shape)  # [4, 512, 224, 224]
        x5 = self.down4(x4)
        # print(x5.shape)  # [4, 512, 224, 224]
        x = self.up4(x5, x4)
        # print(x.shape)  # [4, 256, 224, 224]
        x = self.up3(x, x3)
        # print(x.shape)  # [4, 128, 224, 224]
        x = self.up2(x, x2)
        # print(x.shape)  # [4, 64, 224, 224]
        x = self.up1(x, x1)
        # print(x.shape)  # [4, 64, 224, 224]
        if self.last_activation is not None:
            logits = self.last_activation(self.outc(x))
            # print(logits.shape)  # [4, 1, 224, 224]
            # print("111")
        else:
            logits = self.outc(x)
            # print(logits.shape)
            # print("222")
        # logits = self.outc(x) # if using BCEWithLogitsLoss
        # print(logits.size())
        return logits


if __name__ == "__main__":
    net = UNet_base()
    print(net)
```

```python
UNet_base(
  (inc): ConvBatchNorm(
    (conv): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (activation): ReLU()
  )
  (down1): DownBlock(
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (down2): DownBlock(
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (down3): DownBlock(
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (down4): DownBlock(
    (maxpool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (up4): UpBlock(
    (up): ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (up3): UpBlock(
    (up): ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2))
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(512, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (up2): UpBlock(
    (up): ConvTranspose2d(128, 128, kernel_size=(2, 2), stride=(2, 2))
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(256, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (up1): UpBlock(
    (up): ConvTranspose2d(64, 64, kernel_size=(2, 2), stride=(2, 2))
    (nConvs): Sequential(
      (0): ConvBatchNorm(
        (conv): Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
      (1): ConvBatchNorm(
        (conv): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (activation): ReLU()
      )
    )
  )
  (outc): Conv2d(64, 10, kernel_size=(1, 1), stride=(1, 1))
)
```

