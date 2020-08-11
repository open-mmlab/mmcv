## CUDA ops

We implement common CUDA ops used in detection, segmentation, etc.

- BBoxOverlaps
- CARAFE
- CrissCrossAttention
- ContextBlock
- CornerPool
- Deformable Convolution v1/v2
- Deformable RoIPool
- GeneralizedAttention
- MaskedConv
- NMS
- PSAMask
- RoIPool
- RoIAlign
- SimpleRoIAlign
- SigmoidFocalLoss
- SoftmaxFocalLoss
- SoftNMS
- Synchronized BatchNorm
- Weight standardization


## Troubleshooting

Here are some common issues that you might meet related with ops.

###

```
RuntimeError: CUDA error: invalid configuration argument
```

This error may be due to your poor GPU. Decrease the value of [THREADS_PER_BLOCK](https://github.com/open-mmlab/mmcv/blob/cac22f8cf5a904477e3b5461b1cc36856c2793da/mmcv/ops/csrc/common_cuda_helper.hpp#L10) and recompile mmcv. You can refer [issue#255](https://github.com/open-mmlab/mmcv/issues/419) for details.
