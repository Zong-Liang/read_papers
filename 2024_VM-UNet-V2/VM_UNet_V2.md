

```pyth
VMUNet(
  (vmunet): VSSM(
    (patch_embed): PatchEmbed2D(
      (proj): Conv2d(2, 96, kernel_size=(4, 4), stride=(4, 4))
      (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
    )
    (pos_drop): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.0)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.02857142873108387)
          )
        )
        (downsample): PatchMerging2D(
          (reduction): Linear(in_features=384, out_features=192, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (1): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.05714285746216774)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.08571428805589676)
          )
        )
        (downsample): PatchMerging2D(
          (reduction): Linear(in_features=768, out_features=384, bias=False)
          (norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.11428571492433548)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.1428571492433548)
          )
        )
        (downsample): PatchMerging2D(
          (reduction): Linear(in_features=1536, out_features=768, bias=False)
          (norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): VSSLayer(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.17142857611179352)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.20000000298023224)
          )
        )
      )
    )
    (layers_up): ModuleList(
      (0): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.20000000298023224)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=768, out_features=3072, bias=False)
              (conv2d): Conv2d(1536, 1536, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=1536)
              (act): SiLU()
              (out_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=1536, out_features=768, bias=False)
            )
            (drop_path): timm.DropPath(0.1666666716337204)
          )
        )
      )
      (1): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.13333332538604736)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=384, out_features=1536, bias=False)
              (conv2d): Conv2d(768, 768, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=768)
              (act): SiLU()
              (out_norm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=768, out_features=384, bias=False)
            )
            (drop_path): timm.DropPath(0.09999999403953552)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=768, out_features=1536, bias=False)
          (norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
        )
      )
      (2): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.06666667014360428)
          )
          (1): VSSBlock(
            (ln_1): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=192, out_features=768, bias=False)
              (conv2d): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384)
              (act): SiLU()
              (out_norm): LayerNorm((384,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=384, out_features=192, bias=False)
            )
            (drop_path): timm.DropPath(0.03333333507180214)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=384, out_features=768, bias=False)
          (norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
        )
      )
      (3): VSSLayer_up(
        (blocks): ModuleList(
          (0): VSSBlock(
            (ln_1): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
            (self_attention): SS2D(
              (in_proj): Linear(in_features=96, out_features=384, bias=False)
              (conv2d): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192)
              (act): SiLU()
              (out_norm): LayerNorm((192,), eps=1e-05, elementwise_affine=True)
              (out_proj): Linear(in_features=192, out_features=96, bias=False)
            )
            (drop_path): timm.DropPath(0.0)
          )
        )
        (upsample): PatchExpand2D(
          (expand): Linear(in_features=192, out_features=384, bias=False)
          (norm): LayerNorm((96,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    (final_up): Final_PatchExpand2D(
      (expand): Linear(in_features=96, out_features=384, bias=False)
      (norm): LayerNorm((24,), eps=1e-05, elementwise_affine=True)
    )
    (final_conv): Conv2d(24, 1, kernel_size=(1, 1), stride=(1, 1))
  )
)
```

