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

> code is available at [mmcv/cnn/utils/weight_init.py](../mmcv/cnn/utils/weight_init.py)

During training, a proper initialization strategy is beneficial to speed the
training or obtain a higher performance. In MMCV, we provide some commonly used
methods for initializing modules like `nn.Conv2d`. Of course, we also provide a
high-level APIs for initializing the entire model containing one or more
modules.

#### **Initialization of module**

Initializaing modules, such as `nn.Conv2d`, `nn.Linear` and so on.

We provide the following initialization methods.

- constant_init

  Initialize module parameters with constant values.

    ```python
    >>> import torch.nn as nn
    >>> from mmcv.cnn.utils.weight_init import constant_init
    >>> conv1 = nn.Conv2d(3, 3, 1)
    >>> # constant_init(module, val, bias=0)
    >>> constant_init(conv1, 1, 0)
    ```

- xavier_init

  Initialize module parameters with values according to the method
  described in [Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

    ```python
    >>> import torch.nn as nn
    >>> from mmcv.cnn.utils.weight_init import xavier_init
    >>> conv1 = nn.Conv2d(3, 3, 1)
    >>> # xavier_init(module, gain=1, bias=0, distribution='normal')
    >>> xavier_init(conv1, distribution='normal')
    ```

- normal_init

  Initialize module parameters with the values drawn from a normal distribution.

    ```python
    >>> import torch.nn as nn
    >>> from mmcv.cnn.utils.weight_init import normal_init
    >>> conv1 = nn.Conv2d(3, 3, 1)
    >>> # normal_init(module, mean=0, std=1, bias=0)
    >>> normal_init(conv1, std=0.01, bias=0)
    ```

- uniform_init

  Initialize module parameters with values drawn from a uniform distribution.

    ```python
    >>> import torch.nn as nn
    >>> from mmcv.cnn.utils.weight_init import uniform_init
    >>> conv1 = nn.Conv2d(3, 3, 1)
    >>> # uniform_init(module, a=0, b=1, bias=0)
    >>> uniform_init(conv1, a=0, b=1)
    ```

- kaiming_init

  Initialize module paramters with the valuse according to the method
  described in [Delving deep into rectifiers: Surpassing human-level
  performance on ImageNet classification - He, K. et al. (2015)](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)

    ```python
    >>> import torch.nn as nn
    >>> from mmcv.cnn.utils.weight_init import kaiming_init
    >>> conv1 = nn.Conv2d(3, 3, 1)
    >>> # kaiming_init(module, a=0, mode='fan_out', nonlinearity='relu', bias=0, distribution='normal')
    >>> kaiming_init(conv1)
    ```

- caffe2_xavier_init
  Corresponds to `kaiming_uniform_` in PyTorch.

    ```python
    >>> import torch.nn as nn
    >>> from mmcv.cnn.utils.weight_init import caffe2_xavier_init
    >>> conv1 = nn.Conv2d(3, 3, 1)
    >>> # caffe2_xavier_init(module, bias=0)
    >>> caffe2_xavier_init(conv1)
    ```

- bias_init_with_prob

  Initialize conv/fc bias value according to given probability proposed in [Focal Loss for Dense Object Detection](https://arxiv.org/pdf/1708.02002.pdf).

    ```python
    >>> from mmcv.cnn.utils.weight_init import bias_init_with_prob
    >>> # bias_init_with_prob is proposed in Focal Loss
    >>> bias = bias_init_with_prob(0.01)
    >>> bias
    -4.59511985013459
    ```

#### **Initialization of model**

On the basis of the initialization methods, we define the corresponding initialization classes and register them to `INITIALIZERS`, so we can
use the configuration to initialize the model.

We provide the following initialization classes.

- BaseInit
- BaseInit
- XavierInit
- NormalInit
- UniformInit
- KaimingInit
- Caffe2XavierInit
- PretrainedInit

Before we go deeper into the usage of `initialize`, briefly introducing the
design principle of it is helpful.

- If we don't define `layer` key or `override` key, it will not initialize anything.
- If we define `override` but don't define `layer`, it will initialize parameters with the attribute name in `override`.
- If we only define `layer`, it just initialize the layer in `layer` key.
- If we define `override` and `layer`, `override` has higher priority and will override initialization mechanism.

Now, it is time to introduce the usage of `initialize` in detail.

- Initialize whole module with the same configuration

  Define `layer` for initializing layer with same configuration.

    ```python
    import torch.nn as nn
    from mmcv.cnn.utils.weight_init import initialize

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
    ```

- Initialize specific layer with different configurations

  Define `layer` for initializing layer with different configurations.

    ```python
    import torch.nn as nn
    from mmcv.cnn.utils.weight_init import initialize

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
    ```

- Initialize module with the attribute name

  Define `override` for initializing module with the attribute name.

    ```python
    import torch.nn as nn
    from mmcv.cnn.utils.weight_init import initialize

    class FooNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.feat = nn.Conv1d(3, 1, 3)
            self.reg = nn.Conv2d(3, 3, 3)
            self.cls = nn.Sequential(nn.Conv1d(3, 1, 3), nn.Linear(1,2))

    model = FooNet()
    init_cfg = dict(type='Constant', val=1, bias=2, layer=['Conv1d','Conv2d'],
                    override=dict(type='Constant', name='reg', val=3, bias=4))
    # self.feat and self.cls will be initialized with dict(type='Constant', val=1, bias=2)
    # The module called 'reg' will be initialized with dict(type='Constant', val=3, bias=4)
    initialize(model, init_cfg)
    ```

- Initialize weights with the pretrained model

    ```python
    import torch.nn as nn
    import torchvision.models as models
    from mmcv.cnn.utils.weight_init import initialize

    # initialize weights with the whole model
    model = models.resnet50()
    init_cfg = dict(type='Pretrained',
                    checkpoint='torchvision://resnet50')
    initialize(model, init_cfg)

    # initialize weights of a sub-module with the specific part of a pretrained model by using 'prefix'
    model = models.resnet50()
    url = 'http://download.openmmlab.com/mmdetection/v2.0/retinanet/'\
          'retinanet_r50_fpn_1x_coco/'\
          'retinanet_r50_fpn_1x_coco_20200130-c2398f9e.pth'
    init_cfg = dict(type='Pretrained',
                    checkpoint=url, prefix='backbone.')
    initialize(model, init_cfg)
    ```

- Initialize models inherited from BaseModule, Sequential, ModuleList

  `BaseModule` is inherited from `torch.nn.Module`, and the only different between them is that `BaseModule` implements `init_weight`.

  `Sequential` is inhertied from `BaseModule` and `torch.nn.Sequential`.

  `ModuleList` is inhertied from `BaseModule` and `torch.nn.ModuleList`.

    ```python
    import torch.nn as nn
    from mmcv.runner.base_module import BaseModule, Sequential, ModuleList

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
    model.init_weight()

    # Sequential
    init_cfg1 = dict(type='Constant', layer='Conv1d', val=0., bias=1.)
    init_cfg2 = dict(type='Constant', layer='Conv2d', val=2., bias=3.)
    model1 = FooConv1d(init_cfg1)
    model2 = FooConv2d(init_cfg2)
    seq_model = Sequential(model1, model2)
    seq_model.init_weight()
    # inner init_cfg has highter priority
    model1 = FooConv1d(init_cfg1)
    model2 = FooConv2d(init_cfg2)
    init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.)
    seq_model = Sequential(model1, model2, init_cfg=init_cfg)
    seq_model.init_weight()

    # ModuleList
    model1 = FooConv1d(init_cfg1)
    model2 = FooConv2d(init_cfg2)
    modellist = ModuleList([model1, model2])
    modellist.init_weight()
    # inner init_cfg has highter priority
    init_cfg = dict(type='Constant', layer=['Conv1d', 'Conv2d'], val=4., bias=5.)
    model1 = FooConv1d(init_cfg1)
    model2 = FooConv2d(init_cfg2)
    modellist = ModuleList(model1, model2, init_cfg=init_cfg)
    modellist.init_weight()
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
- `torchvison://xxx`: The model links in `torchvision.models`.Please refer to [torchvision](https://pytorch.org/docs/stable/torchvision/models.html) for details.
- `open-mmlab://xxx`: The model links or filepath provided in default and additional json files.
