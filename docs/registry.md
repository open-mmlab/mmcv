## Registry

MMCV implements [registry](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) to manage different modules that share similar functionalities, e.g., backbones, head, and necks, in detectors.
Most projects in OpenMMLab use registry to manage modules of datasets and models, such as [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [MMClassification](https://github.com/open-mmlab/mmclassification), [MMEditing](https://github.com/open-mmlab/mmediting), etc.

### What is registry

In MMCV, registry can be regarded as a mapping that maps a class to a string.
These classes contained by a single registry usually have similar APIs but implement different algorithms or support different datasets.
With the registry, users can find and instantiate the class through its corresponding string, and use the instantiated module as they want.
One typical example is the config systems in most OpenMMLab projects, which use the registry to create hooks, runners, models, and datasets, through configs.
The detailed doc could be find [here](https://mmcv.readthedocs.io/en/latest/api.html?highlight=registry#mmcv.utils.Registry).

To manage your modules in the codebase by `Registry`, there are three steps as below.

1. Create a build method (optional, in most cases you can just use the default one).
2. Create a registry.
3. Use this registry to manage the modules.

`build_func` argument of `Registry` is to customize how to instantiate the class instance, the default one is `build_from_cfg` implemented [here](https://mmcv.readthedocs.io/en/latest/api.html?highlight=registry#mmcv.utils.build_from_cfg).

### A Simple Example

Here we show a simple example of using registry to manage modules in a package.
You can find more practical examples in OpenMMLab projects.

Assuming we want to implement a series of Dataset Converter for converting different formats of data to the expected data format.
We create a directory as a package named `converters`.
In the package, we first create a file to implement builders, named `converters/builder.py`, as below

Note: in this example, we demonstrate how to use `build_func` argument to customize the way to build a class instance. In most cases, default one would be sufficent.

```python
from mmcv.utils import Registry

# create a build function
def build_converter(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in registry:
        raise KeyError(f'Unrecognized task type {converter_type}')
    else:
        converter_cls = registry.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter

# create a registry for converters
CONVERTERS = Registry('converter', build_func=build_converter)
```

*Note: similar functions like `build_from_cfg` and `build_model_from_cfg` is already implemented, you may directly use them instead of implementing by yourself.*

Then we can implement different converters in the package. For example, implement `Converter1` in `converters/converter1.py`

```python

from .builder import CONVERTERS


# use the registry to manage the module
@CONVERTERS.register_module()
class Converter1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
```

The key step to use registry for managing the modules is to register the implemented module into the registry `CONVERTERS` through
`@CONVERTERS.register_module()` when you are creating the module. By this way, a mapping between a string and the class is built and maintained by `CONVERTERS` as below

```python
'Converter1' -> <class 'Converter1'>
```

If the module is successfully registered, you can use this converter through configs as

```python
converter_cfg = dict(type='Converter1', a=a_value, b=b_value)
converter = CONVERTERS.build(converter_cfg)
```

## Hierarchy Registry

Hierarchy structure is used for similar registries from different packages.
For example, both [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMClassification](https://github.com/open-mmlab/mmclassification) have `MODEL` registry define as follows:
In MMDetection:

```python
from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS
MODELS = Registry('model', parent=MMCV_MODELS)

@MODELS.register_module()
class NetA(nn.Module):
    def forward(self, x):
        return x
```

In MMClassification:

```python
from mmcv.utils import Registry
from mmcv.cnn import MODELS as MMCV_MODELS
MODELS = Registry('model', parent=MMCV_MODELS)

@MODELS.register_module()
class NetB(nn.Module):
    def forward(self, x):
        return x + 1
```

We could build two net in either MMDetection or MMClassification by:

```python
import mmcls # import mmcls to register models
from mmdet.models import MODELS
net_a = MODELS.build(cfg=dict(type='NetA'))
net_b = MODELS.build(cfg=dict(type='mmcls.NetB'))
```

or

```python
import mmdet # import mmcls to register models
from mmcls.models import MODELS
net_a = MODELS.build(cfg=dict(type='mmdet.NetA'))
net_b = MODELS.build(cfg=dict(type='NetB'))
```

Build them by shared `MODELS` registry in MMCV is also feasible:

```python
from mmcv.cnn import MODELS as MMCV_MODELS
net_a = MMCV_MODELS.build(cfg=dict(type='mmdet.NetA'))
net_b = MMCV_MODELS.build(cfg=dict(type='mmcls.NetB'))
```
