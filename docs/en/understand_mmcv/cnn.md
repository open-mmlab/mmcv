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
from mmcv.cnn import build_conv_layer

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
   from mmengine.registry import MODELS

   @MODELS.register_module()
   class MyUpsample:

       def __init__(self, scale_factor):
           pass

       def forward(self, x):
           pass
   ```

2. Import `MyUpsample` somewhere (e.g., in `__init__.py`) and then use it.

   ```python
   from mmcv.cnn import build_upsample_layer

   cfg = dict(type='MyUpsample', scale_factor=2)
   layer = build_upsample_layer(cfg)
   ```

### Module bundles

We also provide common module bundles to facilitate the network construction.
`ConvModule` is a bundle of convolution, normalization and activation layers,
please refer to the [api](api.html#mmcv.cnn.ConvModule) for details.

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
