## 注册器

MMCV 使用 [注册器](https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/registry.py) 来管理具有相似功能的不同模块, 例如, 检测器中的主干网络、头部、和模型颈部。
在 OpenMMLab 家族中的绝大部分开源项目使用注册器去管理数据集和模型的模块，例如 [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d), [MMClassification](https://github.com/open-mmlab/mmclassification), [MMEditing](https://github.com/open-mmlab/mmediting) 等。

```{note}
在 v1.5.1 版本开始支持注册函数的功能。
```

### 什么是注册器

在MMCV中，注册器可以看作类或函数到字符串的映射。
一个注册器中的类或函数通常有相似的接口，但是可以实现不同的算法或支持不同的数据集。
借助注册器，用户可以通过使用相应的字符串查找类或函数，并根据他们的需要实例化对应模块或调用函数获取结果。
一个典型的案例是，OpenMMLab　中的大部分开源项目的配置系统，这些系统通过配置文件来使用注册器创建钩子、执行器、模型和数据集。
可以在[这里](https://mmcv.readthedocs.io/en/latest/api.html?highlight=registry#mmcv.utils.Registry)找到注册器接口使用文档。

使用 `registry`（注册器）管理代码库中的模型，需要以下三个步骤。

1. 创建一个构建方法（可选，在大多数情况下您可以只使用默认方法）
2. 创建注册器
3. 使用此注册器来管理模块

`Registry`（注册器）的参数 `build_func`（构建函数） 用来自定义如何实例化类的实例或如何调用函数获取结果，默认使用 [这里](https://mmcv.readthedocs.io/en/latest/api.html?highlight=registry#mmcv.utils.build_from_cfg) 实现的`build_from_cfg`。

### 一个简单的例子

这里是一个使用注册器管理包中模块的简单示例。您可以在 OpenMMLab 开源项目中找到更多实例。

假设我们要实现一系列数据集转换器（Dataset Converter），用于将不同格式的数据转换为标准数据格式。我们先创建一个名为converters的目录作为包，在包中我们创建一个文件来实现构建器（builder），命名为converters/builder.py，如下

```python
from mmcv.utils import Registry
# 创建转换器（converter）的注册器（registry）
CONVERTERS = Registry('converter')
```

然后我们在包中可以实现不同的转换器（converter），其可以为类或函数。例如，在 `converters/converter1.py` 中实现 `Converter1`，在 `converters/converter2.py` 中实现 `converter2`。

```python
# converter1.py
from .builder import CONVERTERS

# 使用注册器管理模块
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
def converter2(a, b):
    return Converter1(a, b)
```

使用注册器管理模块的关键步骤是，将实现的模块注册到注册表 `CONVERTERS` 中。通过 `@CONVERTERS.register_module()` 装饰所实现的模块，字符串到类或函数之间的映射就可以由 `CONVERTERS` 构建和维护，如下所示：

通过这种方式，就可以通过 `CONVERTERS` 建立字符串与类或函数之间的映射，如下所示：

```python
'Converter1' -> <class 'Converter1'>
'converter2' -> <function 'converter2'>
```

```{note}
只有模块所在的文件被导入时，注册机制才会被触发，所以您需要在某处导入该文件。更多详情请查看 https://github.com/open-mmlab/mmdetection/issues/5974。
```

如果模块被成功注册了，你可以通过配置文件使用这个转换器（converter），如下所示：

```python
converter1_cfg = dict(type='Converter1', a=a_value, b=b_value)
converter2_cfg = dict(type='converter2', a=a_value, b=b_value)
converter1 = CONVERTERS.build(converter1_cfg)
# returns the calling result
result = CONVERTERS.build(converter2_cfg)
```

### 自定义构建函数

假设我们想自定义 `converters` 的构建流程，我们可以实现一个自定义的 `build_func` （构建函数）并将其传递到注册器中。

```python
from mmcv.utils import Registry

# 创建一个构建函数
def build_converter(cfg, registry, *args, **kwargs):
    cfg_ = cfg.copy()
    converter_type = cfg_.pop('type')
    if converter_type not in registry:
        raise KeyError(f'Unrecognized converter type {converter_type}')
    else:
        converter_cls = registry.get(converter_type)

    converter = converter_cls(*args, **kwargs, **cfg_)
    return converter

# 创建一个用于转换器（converters）的注册器，并传递（registry）``build_converter`` 函数
CONVERTERS = Registry('converter', build_func=build_converter)
```

```{note}
注：在这个例子中，我们演示了如何使用参数：`build_func` 自定义构建类的实例的方法。
该功能类似于默认的`build_from_cfg`。在大多数情况下，默认就足够了。
```

`build_model_from_cfg`也实现了在`nn.Sequential`中构建PyTorch模块，你可以直接使用它们。

### 注册器层结构

你也可以从多个 OpenMMLab 开源框架中构建模块，例如，你可以把所有 [MMClassification](https://github.com/open-mmlab/mmclassification) 中的主干网络（backbone）用到 [MMDetection](https://github.com/open-mmlab/mmdetection) 的目标检测中，你也可以融合 [MMDetection](https://github.com/open-mmlab/mmdetection) 中的目标检测模型 和 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) 语义分割模型。

下游代码库中所有 `MODELS` 注册器都是MMCV `MODELS` 注册器的子注册器。基本上，使用以下两种方法从子注册器或相邻兄弟注册器构建模块。

1. 从子注册器中构建

   例如：

   我们在 MMDetection 中定义：

   ```python
   from mmcv.utils import Registry
   from mmcv.cnn import MODELS as MMCV_MODELS
   MODELS = Registry('model', parent=MMCV_MODELS)

   @MODELS.register_module()
   class NetA(nn.Module):
       def forward(self, x):
           return x
   ```

   我们在 MMClassification 中定义：

   ```python
   from mmcv.utils import Registry
   from mmcv.cnn import MODELS as MMCV_MODELS
   MODELS = Registry('model', parent=MMCV_MODELS)

   @MODELS.register_module()
   class NetB(nn.Module):
       def forward(self, x):
           return x + 1
   ```

   我们可以通过以下代码在 MMDetection 或 MMClassification 中构建两个网络：

   ```python
   from mmdet.models import MODELS
   net_a = MODELS.build(cfg=dict(type='NetA'))
   net_b = MODELS.build(cfg=dict(type='mmcls.NetB'))
   ```

   或

   ```python
   from mmcls.models import MODELS
   net_a = MODELS.build(cfg=dict(type='mmdet.NetA'))
   net_b = MODELS.build(cfg=dict(type='NetB'))
   ```

2. 从父注册器中构建

   MMCV中的共享`MODELS`注册器是所有下游代码库的父注册器（根注册器）：

   ```python
   from mmcv.cnn import MODELS as MMCV_MODELS
   net_a = MMCV_MODELS.build(cfg=dict(type='mmdet.NetA'))
   net_b = MMCV_MODELS.build(cfg=dict(type='mmcls.NetB'))
   ```
