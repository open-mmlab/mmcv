## Registry

MMCV implements [registry](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) to manage different modules that share similar functionalities, e.g., backbones, head, and necks, in detectors.
Most projects in OpenMMLab use registry to manage modules of datasets and models, such as [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [MMClassification](https://github.com/open-mmlab/mmclassification), [MMEditing](https://github.com/open-mmlab/mmediting), etc.

```{note}
In v1.5.1 and later, the Registry supports registering functions and calling them.
```

### What is registry

In MMCV, registry can be regarded as a mapping that maps a class or function to a string.
These classes or functions contained by a single registry usually have similar APIs but implement different algorithms or support different datasets.
With the registry, users can find the class or function through its corresponding string, and instantiate the corresponding module or call the function to obtain the result according to needs.
One typical example is the config systems in most OpenMMLab projects, which use the registry to create hooks, runners, models, and datasets, through configs.
The API reference could be found [here](https://mmcv.readthedocs.io/en/latest/api.html?highlight=registry#mmcv.utils.Registry).

To manage your modules in the codebase by `Registry`, there are three steps as below.

1. Create a build method (optional, in most cases you can just use the default one).
2. Create a registry.
3. Use this registry to manage the modules.

`build_func` argument of `Registry` is to customize how to instantiate the class instance or how to call the function to obtain the result, the default one is `build_from_cfg` implemented [here](https://mmcv.readthedocs.io/en/latest/api.html?highlight=registry#mmcv.utils.build_from_cfg).

### A Simple Example

Here we show a simple example of using registry to manage modules in a package.
You can find more practical examples in OpenMMLab projects.

Assuming we want to implement a series of Dataset Converter for converting different formats of data to the expected data format.
We create a directory as a package named `converters`.
In the package, we first create a file to implement builders, named `converters/builder.py`, as below

```python
from mmcv.utils import Registry
# create a registry for converters
CONVERTERS = Registry('converters')
```

Then we can implement different converters that is class or function in the package. For example, implement `Converter1` in `converters/converter1.py`, and `converter2` in `converters/converter2.py`.

```python

from .builder import CONVERTERS

# use the registry to manage the module
@CONVERTERS.register_module()
class Converter1(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
```

```python
# converter2.py
from .builder import CONVERTERS
from .converter1 import Converter1

# 使用注册器管理模块
@CONVERTERS.register_module()
def converter2(a, b)
    return Converter1(a, b)
```

The key step to use registry for managing the modules is to register the implemented module into the registry `CONVERTERS` through
`@CONVERTERS.register_module()` when you are creating the module. By this way, a mapping between a string and the class (function) is built and maintained by `CONVERTERS` as below

```python
'Converter1' -> <class 'Converter1'>
'converter2' -> <function 'converter2'>
```

```{note}
The registry mechanism will be triggered only when the file where the module is located is imported.
So you need to import that file somewhere. More details can be found at https://github.com/open-mmlab/mmdetection/issues/5974.
```

If the module is successfully registered, you can use this converter through configs as

```python
converter1_cfg = dict(type='Converter1', a=a_value, b=b_value)
converter2_cfg = dict(type='converter2', a=a_value, b=b_value)
converter1 = CONVERTERS.build(converter1_cfg)
# returns the calling result
result = CONVERTERS.build(converter2_cfg)
```

### Customize Build Function

Suppose we would like to customize how `converters` are built, we could implement a customized `build_func` and pass it into the registry.

```python
from mmcv.utils import Registry

# create a build function
def build_converter(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in registry:
        raise KeyError(f'Unrecognized converter type {converter_type}')
    else:
        converter_cls = registry.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter

# create a registry for converters and pass ``build_converter`` function
CONVERTERS = Registry('converter', build_func=build_converter)
```

```{note}
In this example, we demonstrate how to use the `build_func` argument to customize the way to build a class instance.
The functionality is similar to the default `build_from_cfg`. In most cases, default one would be sufficient.
`build_model_from_cfg` is also implemented to build PyTorch module in `nn.Sequential`, you may directly use them instead of implementing by yourself.
```

### Hierarchy Registry

You could also build modules from more than one OpenMMLab frameworks, e.g. you could use all backbones in [MMClassification](https://github.com/open-mmlab/mmclassification) for object detectors in [MMDetection](https://github.com/open-mmlab/mmdetection), you may also combine an object detection model in [MMDetection](https://github.com/open-mmlab/mmdetection) and semantic segmentation model in [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).

All `MODELS` registries of downstream codebases are children registries of MMCV's `MODELS` registry.
Basically, there are two ways to build a module from child or sibling registries.

1. Build from children registries.

   For example:

   In MMDetection we define:

   ```python
   from mmcv.utils import Registry
   from mmcv.cnn import MODELS as MMCV_MODELS
   MODELS = Registry('model', parent=MMCV_MODELS)

   @MODELS.register_module()
   class NetA(nn.Module):
       def forward(self, x):
           return x
   ```

   In MMClassification we define:

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
   from mmdet.models import MODELS
   net_a = MODELS.build(cfg=dict(type='NetA'))
   net_b = MODELS.build(cfg=dict(type='mmcls.NetB'))
   ```

   or

   ```python
   from mmcls.models import MODELS
   net_a = MODELS.build(cfg=dict(type='mmdet.NetA'))
   net_b = MODELS.build(cfg=dict(type='NetB'))
   ```

2. Build from parent registry.

   The shared `MODELS` registry in MMCV is the parent registry for all downstream codebases (root registry):

   ```python
   from mmcv.cnn import MODELS as MMCV_MODELS
   net_a = MMCV_MODELS.build(cfg=dict(type='mmdet.NetA'))
   net_b = MMCV_MODELS.build(cfg=dict(type='mmcls.NetB'))
   ```
