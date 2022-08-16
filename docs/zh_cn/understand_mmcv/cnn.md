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

### Model Zoo

除了`torchvision`的预训练模型，我们还提供以下 CNN 的预训练模型：

- VGG Caffe
- ResNet Caffe
- ResNeXt
- ResNet with Group Normalization
- ResNet with Group Normalization and Weight Standardization
- HRNetV2
- Res2Net
- RegNet

#### Model URLs in JSON

MMCV中的Model Zoo Link 由 JSON 文件管理。 json 文件由模型名称及其url或path的键值对组成,一个json文件可能类似于:

```json
{
    "model_a": "https://example.com/models/model_a_9e5bac.pth",
    "model_b": "pretrain/model_b_ab3ef2c.pth"
}
```

可以在[此处](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json)找到托管在 OpenMMLab AWS 上的预训练模型的默认链接。

你可以通过将 `open-mmlab.json` 放在 `MMCV_HOME`下来覆盖默认链接，如果在环境中找不到`MMCV_HOME`，则默认使用 `~/.cache/mmcv`。当然你也可以使用命令 `export MMCV_HOME=/your/path`来设置自己的路径。

外部的json文件将被合并为默认文件，如果相同的键出现在外部`json`和默认`json`中，则将使用外部`json`。

#### Load Checkpoint

`mmcv.load_checkpoint()`的参数`filename`支持以下类型：

- filepath: `checkpoint`路径
- `http://xxx` and `https://xxx`: 下载checkpoint的链接，文件名中必需包含`SHA256`后缀
- `torchvision://xxx`: `torchvision.models`中的模型链接，更多细节参考 [torchvision](https://pytorch.org/docs/stable/torchvision/models.html)
- `open-mmlab://xxx`: 默认和其他 json 文件中提供的模型链接或文件路径
