# UCTransNet

```python
UCTransNet(
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
  (mtc): ChannelTransformer(
    (embeddings_1): Channel_Embeddings(
      (patch_embeddings): Conv2d(64, 64, kernel_size=(16, 16), stride=(16, 16))
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (embeddings_2): Channel_Embeddings(
      (patch_embeddings): Conv2d(128, 128, kernel_size=(8, 8), stride=(8, 8))
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (embeddings_3): Channel_Embeddings(
      (patch_embeddings): Conv2d(256, 256, kernel_size=(4, 4), stride=(4, 4))
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (embeddings_4): Channel_Embeddings(
      (patch_embeddings): Conv2d(512, 512, kernel_size=(2, 2), stride=(2, 2))
      (dropout): Dropout(p=0.1, inplace=False)
    )
    (encoder): Encoder(
      (layer): ModuleList(
        (0): Block_ViT(
          (attn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (attn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (attn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (attn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (attn_norm): LayerNorm((960,), eps=1e-06, elementwise_affine=True)
          (channel_attn): Attention_org(
            (query1): ModuleList(
              (0): Linear(in_features=64, out_features=64, bias=False)
              (1): Linear(in_features=64, out_features=64, bias=False)
              (2): Linear(in_features=64, out_features=64, bias=False)
              (3): Linear(in_features=64, out_features=64, bias=False)
            )
            (query2): ModuleList(
              (0): Linear(in_features=128, out_features=128, bias=False)
              (1): Linear(in_features=128, out_features=128, bias=False)
              (2): Linear(in_features=128, out_features=128, bias=False)
              (3): Linear(in_features=128, out_features=128, bias=False)
            )
            (query3): ModuleList(
              (0): Linear(in_features=256, out_features=256, bias=False)
              (1): Linear(in_features=256, out_features=256, bias=False)
              (2): Linear(in_features=256, out_features=256, bias=False)
              (3): Linear(in_features=256, out_features=256, bias=False)
            )
            (query4): ModuleList(
              (0): Linear(in_features=512, out_features=512, bias=False)
              (1): Linear(in_features=512, out_features=512, bias=False)
              (2): Linear(in_features=512, out_features=512, bias=False)
              (3): Linear(in_features=512, out_features=512, bias=False)
            )
            (key): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (value): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (psi): InstanceNorm2d(4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (softmax): Softmax(dim=3)
            (out1): Linear(in_features=64, out_features=64, bias=False)
            (out2): Linear(in_features=128, out_features=128, bias=False)
            (out3): Linear(in_features=256, out_features=256, bias=False)
            (out4): Linear(in_features=512, out_features=512, bias=False)
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (proj_dropout): Dropout(p=0.1, inplace=False)
          )
          (ffn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (ffn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (ffn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (ffn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (ffn1): Mlp(
            (fc1): Linear(in_features=64, out_features=256, bias=True)
            (fc2): Linear(in_features=256, out_features=64, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn2): Mlp(
            (fc1): Linear(in_features=128, out_features=512, bias=True)
            (fc2): Linear(in_features=512, out_features=128, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn3): Mlp(
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn4): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
        )
        (1): Block_ViT(
          (attn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (attn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (attn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (attn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (attn_norm): LayerNorm((960,), eps=1e-06, elementwise_affine=True)
          (channel_attn): Attention_org(
            (query1): ModuleList(
              (0): Linear(in_features=64, out_features=64, bias=False)
              (1): Linear(in_features=64, out_features=64, bias=False)
              (2): Linear(in_features=64, out_features=64, bias=False)
              (3): Linear(in_features=64, out_features=64, bias=False)
            )
            (query2): ModuleList(
              (0): Linear(in_features=128, out_features=128, bias=False)
              (1): Linear(in_features=128, out_features=128, bias=False)
              (2): Linear(in_features=128, out_features=128, bias=False)
              (3): Linear(in_features=128, out_features=128, bias=False)
            )
            (query3): ModuleList(
              (0): Linear(in_features=256, out_features=256, bias=False)
              (1): Linear(in_features=256, out_features=256, bias=False)
              (2): Linear(in_features=256, out_features=256, bias=False)
              (3): Linear(in_features=256, out_features=256, bias=False)
            )
            (query4): ModuleList(
              (0): Linear(in_features=512, out_features=512, bias=False)
              (1): Linear(in_features=512, out_features=512, bias=False)
              (2): Linear(in_features=512, out_features=512, bias=False)
              (3): Linear(in_features=512, out_features=512, bias=False)
            )
            (key): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (value): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (psi): InstanceNorm2d(4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (softmax): Softmax(dim=3)
            (out1): Linear(in_features=64, out_features=64, bias=False)
            (out2): Linear(in_features=128, out_features=128, bias=False)
            (out3): Linear(in_features=256, out_features=256, bias=False)
            (out4): Linear(in_features=512, out_features=512, bias=False)
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (proj_dropout): Dropout(p=0.1, inplace=False)
          )
          (ffn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (ffn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (ffn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (ffn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (ffn1): Mlp(
            (fc1): Linear(in_features=64, out_features=256, bias=True)
            (fc2): Linear(in_features=256, out_features=64, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn2): Mlp(
            (fc1): Linear(in_features=128, out_features=512, bias=True)
            (fc2): Linear(in_features=512, out_features=128, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn3): Mlp(
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn4): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
        )
        (2): Block_ViT(
          (attn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (attn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (attn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (attn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (attn_norm): LayerNorm((960,), eps=1e-06, elementwise_affine=True)
          (channel_attn): Attention_org(
            (query1): ModuleList(
              (0): Linear(in_features=64, out_features=64, bias=False)
              (1): Linear(in_features=64, out_features=64, bias=False)
              (2): Linear(in_features=64, out_features=64, bias=False)
              (3): Linear(in_features=64, out_features=64, bias=False)
            )
            (query2): ModuleList(
              (0): Linear(in_features=128, out_features=128, bias=False)
              (1): Linear(in_features=128, out_features=128, bias=False)
              (2): Linear(in_features=128, out_features=128, bias=False)
              (3): Linear(in_features=128, out_features=128, bias=False)
            )
            (query3): ModuleList(
              (0): Linear(in_features=256, out_features=256, bias=False)
              (1): Linear(in_features=256, out_features=256, bias=False)
              (2): Linear(in_features=256, out_features=256, bias=False)
              (3): Linear(in_features=256, out_features=256, bias=False)
            )
            (query4): ModuleList(
              (0): Linear(in_features=512, out_features=512, bias=False)
              (1): Linear(in_features=512, out_features=512, bias=False)
              (2): Linear(in_features=512, out_features=512, bias=False)
              (3): Linear(in_features=512, out_features=512, bias=False)
            )
            (key): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (value): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (psi): InstanceNorm2d(4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (softmax): Softmax(dim=3)
            (out1): Linear(in_features=64, out_features=64, bias=False)
            (out2): Linear(in_features=128, out_features=128, bias=False)
            (out3): Linear(in_features=256, out_features=256, bias=False)
            (out4): Linear(in_features=512, out_features=512, bias=False)
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (proj_dropout): Dropout(p=0.1, inplace=False)
          )
          (ffn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (ffn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (ffn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (ffn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (ffn1): Mlp(
            (fc1): Linear(in_features=64, out_features=256, bias=True)
            (fc2): Linear(in_features=256, out_features=64, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn2): Mlp(
            (fc1): Linear(in_features=128, out_features=512, bias=True)
            (fc2): Linear(in_features=512, out_features=128, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn3): Mlp(
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn4): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
        )
        (3): Block_ViT(
          (attn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (attn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (attn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (attn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (attn_norm): LayerNorm((960,), eps=1e-06, elementwise_affine=True)
          (channel_attn): Attention_org(
            (query1): ModuleList(
              (0): Linear(in_features=64, out_features=64, bias=False)
              (1): Linear(in_features=64, out_features=64, bias=False)
              (2): Linear(in_features=64, out_features=64, bias=False)
              (3): Linear(in_features=64, out_features=64, bias=False)
            )
            (query2): ModuleList(
              (0): Linear(in_features=128, out_features=128, bias=False)
              (1): Linear(in_features=128, out_features=128, bias=False)
              (2): Linear(in_features=128, out_features=128, bias=False)
              (3): Linear(in_features=128, out_features=128, bias=False)
            )
            (query3): ModuleList(
              (0): Linear(in_features=256, out_features=256, bias=False)
              (1): Linear(in_features=256, out_features=256, bias=False)
              (2): Linear(in_features=256, out_features=256, bias=False)
              (3): Linear(in_features=256, out_features=256, bias=False)
            )
            (query4): ModuleList(
              (0): Linear(in_features=512, out_features=512, bias=False)
              (1): Linear(in_features=512, out_features=512, bias=False)
              (2): Linear(in_features=512, out_features=512, bias=False)
              (3): Linear(in_features=512, out_features=512, bias=False)
            )
            (key): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (value): ModuleList(
              (0): Linear(in_features=960, out_features=960, bias=False)
              (1): Linear(in_features=960, out_features=960, bias=False)
              (2): Linear(in_features=960, out_features=960, bias=False)
              (3): Linear(in_features=960, out_features=960, bias=False)
            )
            (psi): InstanceNorm2d(4, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)
            (softmax): Softmax(dim=3)
            (out1): Linear(in_features=64, out_features=64, bias=False)
            (out2): Linear(in_features=128, out_features=128, bias=False)
            (out3): Linear(in_features=256, out_features=256, bias=False)
            (out4): Linear(in_features=512, out_features=512, bias=False)
            (attn_dropout): Dropout(p=0.1, inplace=False)
            (proj_dropout): Dropout(p=0.1, inplace=False)
          )
          (ffn_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
          (ffn_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
          (ffn_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
          (ffn_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
          (ffn1): Mlp(
            (fc1): Linear(in_features=64, out_features=256, bias=True)
            (fc2): Linear(in_features=256, out_features=64, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn2): Mlp(
            (fc1): Linear(in_features=128, out_features=512, bias=True)
            (fc2): Linear(in_features=512, out_features=128, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn3): Mlp(
            (fc1): Linear(in_features=256, out_features=1024, bias=True)
            (fc2): Linear(in_features=1024, out_features=256, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
          (ffn4): Mlp(
            (fc1): Linear(in_features=512, out_features=2048, bias=True)
            (fc2): Linear(in_features=2048, out_features=512, bias=True)
            (act_fn): GELU(approximate=none)
            (dropout): Dropout(p=0, inplace=False)
          )
        )
      )
      (encoder_norm1): LayerNorm((64,), eps=1e-06, elementwise_affine=True)
      (encoder_norm2): LayerNorm((128,), eps=1e-06, elementwise_affine=True)
      (encoder_norm3): LayerNorm((256,), eps=1e-06, elementwise_affine=True)
      (encoder_norm4): LayerNorm((512,), eps=1e-06, elementwise_affine=True)
    )
    (reconstruct_1): Reconstruct(
      (conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      (norm): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU(inplace=True)
    )
    (reconstruct_2): Reconstruct(
      (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
      (norm): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU(inplace=True)
    )
    (reconstruct_3): Reconstruct(
      (conv): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1))
      (norm): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU(inplace=True)
    )
    (reconstruct_4): Reconstruct(
      (conv): Conv2d(512, 512, kernel_size=(1, 1), stride=(1, 1))
      (norm): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation): ReLU(inplace=True)
    )
  )
  (up4): UpBlock_attention(
    (up): Upsample(scale_factor=2.0, mode=nearest)
    (coatt): CCA(
      (mlp_x): Sequential(
        (0): Flatten()
        (1): Linear(in_features=512, out_features=512, bias=True)
      )
      (mlp_g): Sequential(
        (0): Flatten()
        (1): Linear(in_features=512, out_features=512, bias=True)
      )
      (relu): ReLU(inplace=True)
    )
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
  (up3): UpBlock_attention(
    (up): Upsample(scale_factor=2.0, mode=nearest)
    (coatt): CCA(
      (mlp_x): Sequential(
        (0): Flatten()
        (1): Linear(in_features=256, out_features=256, bias=True)
      )
      (mlp_g): Sequential(
        (0): Flatten()
        (1): Linear(in_features=256, out_features=256, bias=True)
      )
      (relu): ReLU(inplace=True)
    )
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
  (up2): UpBlock_attention(
    (up): Upsample(scale_factor=2.0, mode=nearest)
    (coatt): CCA(
      (mlp_x): Sequential(
        (0): Flatten()
        (1): Linear(in_features=128, out_features=128, bias=True)
      )
      (mlp_g): Sequential(
        (0): Flatten()
        (1): Linear(in_features=128, out_features=128, bias=True)
      )
      (relu): ReLU(inplace=True)
    )
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
  (up1): UpBlock_attention(
    (up): Upsample(scale_factor=2.0, mode=nearest)
    (coatt): CCA(
      (mlp_x): Sequential(
        (0): Flatten()
        (1): Linear(in_features=64, out_features=64, bias=True)
      )
      (mlp_g): Sequential(
        (0): Flatten()
        (1): Linear(in_features=64, out_features=64, bias=True)
      )
      (relu): ReLU(inplace=True)
    )
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
  (outc): Conv2d(64, 1, kernel_size=(1, 1), stride=(1, 1))
  (last_activation): Sigmoid()
)
```

