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

We wrap some initialization methods which accept a module as argument.

- `constant_init`
- `xavier_init`
- `normal_init`
- `uniform_init`
- `kaiming_init`
- `caffe2_xavier_init`
- `bias_init_with_prob`

Examples:

```python
conv1 = nn.Conv2d(3, 3, 1)
normal_init(conv1, std=0.01, bias=0)
xavier_init(conv1, distribution='uniform')
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
