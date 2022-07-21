## 卷积神经网络

我们为卷积神经网络提供了一些构建模块，包括层构建、模块组件和权重初始化。

### 网络层的构建

在运行实验时，我们可能需要尝试同属一种类型但不同配置的层，但又不希望每次都修改代码。于是我们提供一些层构建方法，可以从字典构建层，字典可以在配置文件中配置，也可以通过命令行参数指定。

#### 用法

一个简单的例子：

```python
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
   from mmcv.cnn import UPSAMPLE_LAYERS

   @UPSAMPLE_LAYERS.register_module()
   class MyUpsample:

       def __init__(self, scale_factor):
           pass

       def forward(self, x):
           pass
   ```

2. 在某处导入 `MyUpsample` （例如 `__init__.py` ）然后使用它：

   ```python
   cfg = dict(type='MyUpsample', scale_factor=2)
   layer = build_upsample_layer(cfg)
   ```

### 模块组件

我们还提供了常用的模块组件，以方便网络构建。
卷积组件 `ConvModule` 由 convolution、normalization以及activation layers 组成，更多细节请参考 [ConvModule api](api.html#mmcv.cnn.ConvModule)。

```python
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

### Weight initialization

> 实现细节可以在 [mmcv/cnn/utils/weight_init.py](../../mmcv/cnn/utils/weight_init.py)中找到

在训练过程中，适当的初始化策略有利于加快训练速度或者获得更高的性能。 在MMCV中，我们提供了一些常用的方法来初始化模块，比如 `nn.Conv2d` 模块。当然，我们也提供了一些高级API，可用于初始化包含一个或多个模块的模型。

#### Initialization functions

以函数的方式初始化 `nn.Module` ，例如 `nn.Conv2d` 、 `nn.Linear` 等。

我们提供以下初始化方法，

- constant_init

  使用给定常量值初始化模型参数

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import constant_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # constant_init(module, val, bias=0)
  >>> constant_init(conv1, 1, 0)
  >>> conv1.weight
  ```

- xavier_init

  按照 [Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf) 描述的方法初始化模型参数

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import xavier_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # xavier_init(module, gain=1, bias=0, distribution='normal')
  >>> xavier_init(conv1, distribution='normal')
  ```

- normal_init

  使用正态分布（高斯分布）初始化模型参数

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import normal_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # normal_init(module, mean=0, std=1, bias=0)
  >>> normal_init(conv1, std=0.01, bias=0)
  ```

- uniform_init

  使用均匀分布初始化模型参数

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import uniform_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # uniform_init(module, a=0, b=1, bias=0)
  >>> uniform_init(conv1, a=0, b=1)
  ```

- kaiming_init

  按照 [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification - He, K. et al. (2015)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf) 描述的方法来初始化模型参数。

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import kaiming_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal')
  >>> kaiming_init(conv1)
  ```

- caffe2_xavier_init

  caffe2中实现的 `xavier initialization`，对应于 PyTorch中的 `kaiming_uniform_`

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import caffe2_xavier_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # caffe2_xavier_init(module, bias=0)
  >>> caffe2_xavier_init(conv1)
  ```

- bias_init_with_prob

  根据给定的概率初始化 `conv/fc`, 这在 [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf) 提出。

  ```python
  >>> from mmcv.cnn import bias_init_with_prob
  >>> # bias_init_with_prob is proposed in Focal Loss
  >>> bias = bias_init_with_prob(0.01)
  >>> bias
  -4.59511985013459
  ```

#### Initializers and configs

在初始化方法的基础上，我们定义了相应的初始化类，并将它们注册到 `INITIALIZERS` 中，这样我们就可以使用 `config` 配置来初始化模型了。

我们提供以下初始化类：

- ConstantInit
- XavierInit
- NormalInit
- UniformInit
- KaimingInit
- Caffe2XavierInit
- PretrainedInit

接下来详细介绍 `initialize` 的使用方法

1. 通过关键字 `layer` 来初始化模型

   如果我们只定义了关键字 `layer` ，那么只初始化 `layer` 中包含的层。

   注意: 关键字 `layer` 支持的模块是带有 weights 和 bias 属性的 PyTorch 模块，所以不支持 `MultiheadAttention layer`

- 定义关键字 `layer` 列表并使用相同相同配置初始化模块

  ```python
  import torch.nn as nn
  from mmcv.cnn import initialize

  class FooNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.feat = nn.Conv1d(3, 1, 3)
          self.reg = nn.Conv2d(3, 3, 3)
          self.cls = nn.Linear(1, 2)

  model = FooNet()
  init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d', 'Linear'], val=1)
  # 使用相同的配置初始化整个模块
  initialize(model, init_cfg)
  # model.feat.weight
  # Parameter containing:
  # tensor([[[1., 1., 1.],
  #          [1., 1., 1.],
  #          [1., 1., 1.]]], requires_grad=True)
  ```

- 定义关键字 `layer` 用于初始化不同配置的层

  ```python
  import torch.nn as nn
  from mmcv.cnn.utils import initialize

  class FooNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.feat = nn.Conv1d(3, 1, 3)
          self.reg = nn.Conv2d(3, 3, 3)
          self.cls = nn.Linear(1,2)

  model = FooNet()
  init_cfg = [dict(type='Constant', layer='Conv1d', val=1),
              dict(type='Constant', layer='Conv2d', val=2),
              dict(type='Constant', layer='Linear', val=3)]
  # nn.Conv1d 使用 dict(type='Constant', val=1) 初始化
  # nn.Conv2d 使用 dict(type='Constant', val=2) 初始化
  # nn.Linear 使用 dict(type='Constant', val=3) 初始化
  initialize(model, init_cfg)
  # model.reg.weight
  # Parameter containing:
  # tensor([[[[2., 2., 2.],
  #           [2., 2., 2.],
  #           [2., 2., 2.]],
  #          ...,
  #          [[2., 2., 2.],
  #           [2., 2., 2.],
  #           [2., 2., 2.]]]], requires_grad=True)
  ```

2. 定义关键字`override`初始化模型

- 当用属性名初始化某个特定部分时, 我们可以使用关键字 `override`, 关键字 `override` 对应的Value会替代init_cfg中相应的值

  ```python
  import torch.nn as nn
  from mmcv.cnn import initialize

  class FooNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.feat = nn.Conv1d(3, 1, 3)
          self.reg = nn.Conv2d(3, 3, 3)
          self.cls = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))

  # 如果我们想将模型的权重初始化为 1，将偏差初始化为 2
  # 但希望 `reg` 中的权重为 3，偏差为 4，则我们可以使用关键字override

  model = FooNet()
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'], val=1, bias=2,
                  override=dict(type='Constant', name='reg', val=3, bias=4))
  #  使用 dict(type='Constant', val=1, bias=2)来初始化 self.feat and self.cls
  # 使用dict(type='Constant', val=3, bias=4)来初始化‘reg’模块。
  initialize(model, init_cfg)
  # model.reg.weight
  # Parameter containing:
  # tensor([[[[3., 3., 3.],
  #           [3., 3., 3.],
  #           [3., 3., 3.]],
  #           ...,
  #           [[3., 3., 3.],
  #            [3., 3., 3.],
  #            [3., 3., 3.]]]], requires_grad=True)
  ```

- 如果 init_cfg 中的关键字`layer`为None，则只初始化在关键字override中的子模块，并且省略override中的 type 和其他参数

  ```python
  model = FooNet()
  init_cfg = dict(type='Constant', val=1, bias=2, override=dict(name='reg'))
  # self.feat 和 self.cls 使用pyTorch默认的初始化
  # 将使用 dict(type='Constant', val=1, bias=2) 初始化名为 'reg' 的模块
  initialize(model, init_cfg)
  # model.reg.weight
  # Parameter containing:
  # tensor([[[[1., 1., 1.],
  #           [1., 1., 1.],
  #           [1., 1., 1.]],
  #           ...,
  #           [[1., 1., 1.],
  #            [1., 1., 1.],
  #            [1., 1., 1.]]]], requires_grad=True)
  ```

- 如果我们没有定义关键字`layer`或`override` , 将不会初始化任何东西

- 关键字`override`的无效用法

  ```python
  # 没有重写任何子模块
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'],
                  val=1, bias=2,
                  override=dict(type='Constant', val=3, bias=4))

  # 没有指定type，即便有其他参数，也是无效的。
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'],
                  val=1, bias=2,
                  override=dict(name='reg', val=3, bias=4))
  ```

3. 用预训练模型初始化

   ```python
   import torch.nn as nn
   import torchvision.models as models
   from mmcv.cnn import initialize

   # 使用预训练模型来初始化
   model = models.resnet50()
   # model.conv1.weight
   # Parameter containing:
   # tensor([[[[-6.7435e-03, -2.3531e-02, -9.0143e-03,  ..., -2.1245e-03,
   #            -1.8077e-03,  3.0338e-03],
   #           [-1.2603e-02, -2.7831e-02,  2.3187e-02,  ..., -1.5793e-02,
   #             1.1655e-02,  4.5889e-03],
   #           [-3.7916e-02,  1.2014e-02,  1.3815e-02,  ..., -4.2651e-03,
   #             1.7314e-02, -9.9998e-03],
   #           ...,

   init_cfg = dict(type='Pretrained',
                   checkpoint='torchvision://resnet50')
   initialize(model, init_cfg)
   # model.conv1.weight
   # Parameter containing:
   # tensor([[[[ 1.3335e-02,  1.4664e-02, -1.5351e-02,  ..., -4.0896e-02,
   #            -4.3034e-02, -7.0755e-02],
   #           [ 4.1205e-03,  5.8477e-03,  1.4948e-02,  ...,  2.2060e-03,
   #            -2.0912e-02, -3.8517e-02],
   #           [ 2.2331e-02,  2.3595e-02,  1.6120e-02,  ...,  1.0281e-01,
   #             6.2641e-02,  5.1977e-02],
   #           ...,

   # 使用关键字'prefix'用预训练模型的特定部分来初始化子模块权重
   model = models.resnet50()
   url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
         'retinanet_r50_fpn_1x_coco/'\
         'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
   init_cfg = dict(type='Pretrained',
                   checkpoint=url, prefix='backbone.')
   initialize(model, init_cfg)
   ```

4. 初始化继承自BaseModule、Sequential、ModuleList、ModuleDict的模型

   `BaseModule` 继承自 `torch.nn.Module`, 它们之间唯一的不同是 `BaseModule` 实现了 `init_weight`

   `Sequential` 继承自 `BaseModule` 和 `torch.nn.Sequential`

   `ModuleList` 继承自 `BaseModule` 和 `torch.nn.ModuleList`

   `ModuleDict` 继承自 `BaseModule` 和 `torch.nn.ModuleDict`

   ```python
   import torch.nn as nn
   from mmcv.runner import BaseModule, Sequential, ModuleList, ModuleDict

   class FooConv1d(BaseModule):

       def __init__(self, init_cfg=None):
           super().__init__(init_cfg)
           self.conv1d = nn.Conv1d(4, 1, 4)

       def forward(self, x):
           return self.conv1d(x)

   class FooConv2d(BaseModule):

       def __init__(self, init_cfg=None):
           super().__init__(init_cfg)
           self.conv2d = nn.Conv2d(3, 1, 3)

       def forward(self, x):
           return self.conv2d(x)

   # BaseModule
   init_cfg = dict(type='Constant', layer='Conv1d', val=0., bias=1.)
   model = FooConv1d(init_cfg)
   model.init_weights()
   # model.conv1d.weight
   # Parameter containing:
   # tensor([[[0., 0., 0., 0.],
   #        [0., 0., 0., 0.],
   #        [0., 0., 0., 0.],
   #        [0., 0., 0., 0.]]], requires_grad=True)

   # Sequential
   init_cfg1 = dict(type='Constant', layer='Conv1d', val=0., bias=1.)
   init_cfg2 = dict(type='Constant', layer='Conv2d', val=2., bias=3.)
   model1 = FooConv1d(init_cfg1)
   model2 = FooConv2d(init_cfg2)
   seq_model = Sequential(model1, model2)
   seq_model.init_weights()
   # seq_model[0].conv1d.weight
   # Parameter containing:
   # tensor([[[0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.]]], requires_grad=True)
   # seq_model[1].conv2d.weight
   # Parameter containing:
   # tensor([[[[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]],
   #         ...,
   #          [[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]]]], requires_grad=True)

   # inner init_cfg has higher priority
   model1 = FooConv1d(init_cfg1)
   model2 = FooConv2d(init_cfg2)
   init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.)
   seq_model = Sequential(model1, model2, init_cfg=init_cfg)
   seq_model.init_weights()
   # seq_model[0].conv1d.weight
   # Parameter containing:
   # tensor([[[0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.]]], requires_grad=True)
   # seq_model[1].conv2d.weight
   # Parameter containing:
   # tensor([[[[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]],
   #         ...,
   #          [[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]]]], requires_grad=True)

   # ModuleList
   model1 = FooConv1d(init_cfg1)
   model2 = FooConv2d(init_cfg2)
   modellist = ModuleList([model1, model2])
   modellist.init_weights()
   # modellist[0].conv1d.weight
   # Parameter containing:
   # tensor([[[0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.]]], requires_grad=True)
   # modellist[1].conv2d.weight
   # Parameter containing:
   # tensor([[[[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]],
   #         ...,
   #          [[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]]]], requires_grad=True)

   # inner init_cfg has higher priority
   model1 = FooConv1d(init_cfg1)
   model2 = FooConv2d(init_cfg2)
   init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.)
   modellist = ModuleList([model1, model2], init_cfg=init_cfg)
   modellist.init_weights()
   # modellist[0].conv1d.weight
   # Parameter containing:
   # tensor([[[0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.]]], requires_grad=True)
   # modellist[1].conv2d.weight
   # Parameter containing:
   # tensor([[[[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]],
   #         ...,
   #          [[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]]]], requires_grad=True)

   # ModuleDict
   model1 = FooConv1d(init_cfg1)
   model2 = FooConv2d(init_cfg2)
   modeldict = ModuleDict(dict(model1=model1, model2=model2))
   modeldict.init_weights()
   # modeldict['model1'].conv1d.weight
   # Parameter containing:
   # tensor([[[0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.]]], requires_grad=True)
   # modeldict['model2'].conv2d.weight
   # Parameter containing:
   # tensor([[[[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]],
   #         ...,
   #          [[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]]]], requires_grad=True)

   # inner init_cfg has higher priority
   model1 = FooConv1d(init_cfg1)
   model2 = FooConv2d(init_cfg2)
   init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.)
   modeldict = ModuleDict(dict(model1=model1, model2=model2), init_cfg=init_cfg)
   modeldict.init_weights()
   # modeldict['model1'].conv1d.weight
   # Parameter containing:
   # tensor([[[0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.],
   #         [0., 0., 0., 0.]]], requires_grad=True)
   # modeldict['model2'].conv2d.weight
   # Parameter containing:
   # tensor([[[[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]],
   #         ...,
   #          [[2., 2., 2.],
   #           [2., 2., 2.],
   #           [2., 2., 2.]]]], requires_grad=True)
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
