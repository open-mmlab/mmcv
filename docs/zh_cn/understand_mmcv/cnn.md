## 卷积神经网络

我们为卷积神经网络提供了一些构建模块，包括层构建、模块组件和权重初始化。

### 网络层的构建

在运行实验时，我们可能需要尝试同属一种类型但不同配置的层，但又不希望每次都修改代码。于是我们提供一些层构建方法，可以从字典构建层，字典可以在配置文件中配置，也可以通过命令行参数指定。

#### 用法

一个简单的例子：

```python
from mmcv.cnn import build_conv_layer

cfg = dict(type='Conv3d')
layer = build_conv_layer(cfg, in_channels=3, out_channels=8, kernel_size=3)
```

- `build_conv_layer`: 支持的类型包括 Conv1d、Conv2d、Conv3d、Conv (Conv是Conv2d的别名）
- `build_norm_layer`: 支持的类型包括 BN1d、BN2d、BN3d、BN (alias for BN2d)、SyncBN、GN、LN、IN1d、IN2d、IN3d、IN（IN是IN2d的别名）
- `build_activation_layer`：支持的类型包括 ReLU、LeakyReLU、PReLU、RReLU、ReLU6、ELU、Sigmoid、Tanh、GELU
- `build_upsample_layer`: 支持的类型包括 nearest、bilinear、deconv、pixel_shuffle
- `build_padding_layer`: 支持的类型包括 zero、reflect、replicate

#### 拓展

我们还允许自定义层和算子来扩展构建方法。

1. 编写和注册自己的模块：

   ```python
   from mmengine.registry import MODELS

   @MODELS.register_module()
   class MyUpsample:

       def __init__(self, scale_factor):
           pass

       def forward(self, x):
           pass
   ```

2. 在某处导入 `MyUpsample` （例如 `__init__.py` ）然后使用它：

   ```python
   from mmcv.cnn import build_upsample_layer

   cfg = dict(type='MyUpsample', scale_factor=2)
   layer = build_upsample_layer(cfg)
   ```

### 模块组件

我们还提供了常用的模块组件，以方便网络构建。
卷积组件 `ConvModule` 由 convolution、normalization以及activation layers 组成，更多细节请参考 [ConvModule api](api.html#mmcv.cnn.ConvModule)。

```python
from mmcv.cnn import ConvModule

# conv + bn + relu
conv = ConvModule(3, 8, 2, norm_cfg=dict(type='BN'))
# conv + gn + relu
conv = ConvModule(3, 8, 2, norm_cfg=dict(type='GN', num_groups=2))
# conv + relu
conv = ConvModule(3, 8, 2)
# conv
conv = ConvModule(3, 8, 2, act_cfg=None)
# conv + leaky relu
conv = ConvModule(3, 8, 3, padding=1, act_cfg=dict(type='LeakyReLU'))
# bn + conv + relu
conv = ConvModule(
    3, 8, 2, norm_cfg=dict(type='BN'), order=('norm', 'conv', 'act'))
```
