## CNN

We provide some building bricks for CNNs, including layer building, module bundles and weight initialization.

### Layer building

We may need to try different layers of the same type when running experiments,
but do not want to modify the code from time to time.
Here we provide some layer building methods to construct layers from a dict,
which can be written in configs or specified via command line arguments.

#### Usage

A simplest example is

```python
cfg = dict(type='Conv3d')
layer = build_conv_layer(cfg, in_channels=3, out_channels=8, kernel_size=3)
```

- `build_conv_layer`: Supported types are Conv1d, Conv2d, Conv3d, Conv (alias for Conv2d).
- `build_norm_layer`: Supported types are BN1d, BN2d, BN3d, BN (alias for BN2d), SyncBN, GN, LN, IN1d, IN2d, IN3d, IN (alias for IN2d).
- `build_activation_layer`: Supported types are ReLU, LeakyReLU, PReLU, RReLU, ReLU6, ELU, Sigmoid, Tanh, GELU.
- `build_upsample_layer`: Supported types are nearest, bilinear, deconv, pixel_shuffle.
- `build_padding_layer`: Supported types are zero, reflect, replicate.

#### Extension

We also allow extending the building methods with custom layers and operators.

1. Write and register your own module.

   ```python
   from mmcv.cnn import UPSAMPLE_LAYERS

   @UPSAMPLE_LAYERS.register_module()
   class MyUpsample:

       def __init__(self, scale_factor):
           pass

       def forward(self, x):
           pass
   ```

2. Import `MyUpsample` somewhere (e.g., in `__init__.py`) and then use it.

   ```python
   cfg = dict(type='MyUpsample', scale_factor=2)
   layer = build_upsample_layer(cfg)
   ```

### Module bundles

We also provide common module bundles to facilitate the network construction.
`ConvModule` is a bundle of convolution, normalization and activation layers,
please refer to the [api](api.html#mmcv.cnn.ConvModule) for details.

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

> Implementation details are available at [mmcv/cnn/utils/weight_init.py](../../mmcv/cnn/utils/weight_init.py)

During training, a proper initialization strategy is beneficial to speed up the
training or obtain a higher performance. In MMCV, we provide some commonly used
methods for initializing modules like `nn.Conv2d`. Of course, we also provide
high-level APIs for initializing models containing one or more
modules.

#### Initialization functions

Initialize a `nn.Module` such as `nn.Conv2d`, `nn.Linear` in a functional way.

We provide the following initialization methods.

- constant_init

  Initialize module parameters with constant values.

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import constant_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # constant_init(module, val, bias=0)
  >>> constant_init(conv1, 1, 0)
  >>> conv1.weight
  ```

- xavier_init

  Initialize module parameters with values according to the method
  described in [Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import xavier_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # xavier_init(module, gain=1, bias=0, distribution='normal')
  >>> xavier_init(conv1, distribution='normal')
  ```

- normal_init

  Initialize module parameters with the values drawn from a normal distribution.

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import normal_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # normal_init(module, mean=0, std=1, bias=0)
  >>> normal_init(conv1, std=0.01, bias=0)
  ```

- uniform_init

  Initialize module parameters with values drawn from a uniform distribution.

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import uniform_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # uniform_init(module, a=0, b=1, bias=0)
  >>> uniform_init(conv1, a=0, b=1)
  ```

- kaiming_init

  Initialize module parameters with the values according to the method
  described in [Delving deep into rectifiers: Surpassing human-level
  performance on ImageNet classification - He, K. et al. (2015)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import kaiming_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal')
  >>> kaiming_init(conv1)
  ```

- caffe2_xavier_init

  The xavier initialization is implemented in caffe2, which corresponds to `kaiming_uniform_` in PyTorch.

  ```python
  >>> import torch.nn as nn
  >>> from mmcv.cnn import caffe2_xavier_init
  >>> conv1 = nn.Conv2d(3, 3, 1)
  >>> # caffe2_xavier_init(module, bias=0)
  >>> caffe2_xavier_init(conv1)
  ```

- bias_init_with_prob

  Initialize conv/fc bias value according to a given probability, as proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf).

  ```python
  >>> from mmcv.cnn import bias_init_with_prob
  >>> # bias_init_with_prob is proposed in Focal Loss
  >>> bias = bias_init_with_prob(0.01)
  >>> bias
  -4.59511985013459
  ```

#### Initializers and configs

On the basis of the initialization methods, we define the corresponding initialization classes and register them to `INITIALIZERS`, so we can
use the configuration to initialize the model.

We provide the following initialization classes.

- ConstantInit
- XavierInit
- NormalInit
- UniformInit
- KaimingInit
- Caffe2XavierInit
- PretrainedInit

Let us introduce the usage of `initialize` in detail.

1. Initialize model by `layer` key

   If we only define `layer`, it just initialize the layer in `layer` key.

   NOTE: Value of `layer` key is the class name with attributes weights and bias of Pytorch, so `MultiheadAttention layer` is not supported.

- Define `layer` key for initializing module with same configuration.

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
  # initialize whole module with same configuration
  initialize(model, init_cfg)
  # model.feat.weight
  # Parameter containing:
  # tensor([[[1., 1., 1.],
  #          [1., 1., 1.],
  #          [1., 1., 1.]]], requires_grad=True)
  ```

- Define `layer` key for initializing layer with different configurations.

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
  # nn.Conv1d will be initialized with dict(type='Constant', val=1)
  # nn.Conv2d will be initialized with dict(type='Constant', val=2)
  # nn.Linear will be initialized with dict(type='Constant', val=3)
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

2. Initialize model by `override` key

- When initializing some specific part with its attribute name, we can use `override` key, and the value in `override` will ignore the value in init_cfg.

  ```python
  import torch.nn as nn
  from mmcv.cnn import initialize

  class FooNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.feat = nn.Conv1d(3, 1, 3)
          self.reg = nn.Conv2d(3, 3, 3)
          self.cls = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))

  # if we would like to initialize model's weights as 1 and bias as 2
  # but weight in `reg` as 3 and bias 4, we can use override key
  model = FooNet()
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'], val=1, bias=2,
                  override=dict(type='Constant', name='reg', val=3, bias=4))
  # self.feat and self.cls will be initialized with dict(type='Constant', val=1, bias=2)
  # The module called 'reg' will be initialized with dict(type='Constant', val=3, bias=4)
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

- If `layer` is None in init_cfg, only sub-module with the name in override will be initialized, and type and other args in override can be omitted.

  ```python
  model = FooNet()
  init_cfg = dict(type='Constant', val=1, bias=2, override=dict(name='reg'))
  # self.feat and self.cls will be initialized by Pytorch
  # The module called 'reg' will be initialized with dict(type='Constant', val=1, bias=2)
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

- If we don't define `layer` key or `override` key, it will not initialize anything.

- Invalid usage

  ```python
  # It is invalid that override don't have name key
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'],
                  val=1, bias=2,
                  override=dict(type='Constant', val=3, bias=4))

  # It is also invalid that override has name and other args except type
  init_cfg = dict(type='Constant', layer=['Conv1d','Conv2d'],
                  val=1, bias=2,
                  override=dict(name='reg', val=3, bias=4))
  ```

3. Initialize model with the pretrained model

   ```python
   import torch.nn as nn
   import torchvision.models as models
   from mmcv.cnn import initialize

   # initialize model with pretrained model
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

   # initialize weights of a sub-module with the specific part of a pretrained model by using 'prefix'
   model = models.resnet50()
   url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
         'retinanet_r50_fpn_1x_coco/'\
         'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
   init_cfg = dict(type='Pretrained',
                   checkpoint=url, prefix='backbone.')
   initialize(model, init_cfg)
   ```

4. Initialize model inherited from BaseModule, Sequential, ModuleList, ModuleDict

   `BaseModule` is inherited from `torch.nn.Module`, and the only different between them is that `BaseModule` implements `init_weights()`.

   `Sequential` is inherited from `BaseModule` and `torch.nn.Sequential`.

   `ModuleList` is inherited from `BaseModule` and `torch.nn.ModuleList`.

   `ModuleDict` is inherited from `BaseModule` and `torch.nn.ModuleDict`.

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

Besides torchvision pre-trained models, we also provide pre-trained models of following CNN:

- VGG Caffe
- ResNet Caffe
- ResNeXt
- ResNet with Group Normalization
- ResNet with Group Normalization and Weight Standardization
- HRNetV2
- Res2Net
- RegNet

#### Model URLs in JSON

The model zoo links in MMCV are managed by JSON files.
The json file consists of key-value pair of model name and its url or path.
An example json file could be like:

```json
{
    "model_a": "https://example.com/models/model_a_9e5bac.pth",
    "model_b": "pretrain/model_b_ab3ef2c.pth"
}
```

The default links of the pre-trained models hosted on OpenMMLab AWS could be found [here](https://github.com/open-mmlab/mmcv/blob/master/mmcv/model_zoo/open_mmlab.json).

You may override default links by putting `open-mmlab.json` under `MMCV_HOME`. If `MMCV_HOME` is not find in the environment, `~/.cache/mmcv` will be used by default. You may `export MMCV_HOME=/your/path` to use your own path.

The external json files will be merged into default one. If the same key presents in both external json and default json, the external one will be used.

#### Load Checkpoint

The following types are supported for `filename` argument of `mmcv.load_checkpoint()`.

- filepath: The filepath of the checkpoint.
- `http://xxx` and `https://xxx`: The link to download the checkpoint. The `SHA256` postfix should be contained in the filename.
- `torchvision://xxx`: The model links in `torchvision.models`.Please refer to [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) for details.
- `open-mmlab://xxx`: The model links or filepath provided in default and additional json files.
