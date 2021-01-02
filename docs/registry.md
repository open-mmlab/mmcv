## Registry

MMCV implements [registry](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) to manage different modules that share similar functionalities, e.g., backbones, head, and necks, in detectors.
Most projects in OpenMMLab use registry to manage modules of datasets and models, such as [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [MMClassification](https://github.com/open-mmlab/mmclassification), [MMEditing](https://github.com/open-mmlab/mmediting), etc.

### What is registry

In MMCV, registry can be regarded as a mapping that maps a class to a string.
These classes contained by a single registry usually have similar APIs but implement different algorithms or support different datasets.
With the registry, users can find and instantiate the class through its corresponding string, and use the instantiated module as they want.
One typical example is the config systems in most OpenMMLab projects, which use the registry to create hooks, runners, models, and datasets, through configs.

To manage your modules in the codebase by `Registry`, there are three steps as below.

1. Create an registry
2. Create a build method
3. Use this registry to manage the modules

### A Simple Example

Here we show a simple example of using registry to manage modules in a package.
You can find more practical examples in OpenMMLab projects.

Assuming we want to implement a series of Dataset Converter for converting different formats of data to the expected data format.
We create directory as a package named `converters`.
In the package, we first create a file to implement builders, named `converters/builder.py`, as below

```python
from mmcv.utils import Registry

# create a registry for converters
CONVERTERS = Registry('converter')


# create a build function
def build_converter(cfg, *args, **kwargs):
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in CONVERTERS:
        raise KeyError(f'Unrecognized task type {converter_type}')
    else:
        converter_cls = CONVERTERS.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter
```

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
converter = build_converter(converter_cfg)
```
