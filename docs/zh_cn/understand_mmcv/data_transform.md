# 数据变换

在 OpenMMLab 算法库中，数据集的构建和数据的准备是相互解耦的。通常，数据集的构建只对数据集进行解析，记录每个样本的基本信息；而数据的准备则是通过一系列的数据变换，根据样本的基本信息进行数据加载、预处理、格式化等操作。

## 数据变换的设计

在 MMCV 中，我们使用各种可调用的数据变换类来进行数据的操作。这些数据变换类可以接受若干配置参数进行实例化，之后通过调用的方式对输入的数据字典进行处理。同时，我们约定所有数据变换都接受一个字典作为输入，并将处理后的数据输出为一个字典。一个简单的例子如下：

```python
>>> import numpy as np
>>> from mmcv.transforms import Resize
>>>
>>> transform = Resize(scale=(224, 224))
>>> data_dict = {'img': np.random.rand(256, 256, 3)}
>>> data_dict = transform(data_dict)
>>> print(data_dict['img'].shape)
(224, 224, 3)
```

数据变换类会读取输入字典的某些字段，并且可能添加、或者更新某些字段。这些字段的键大部分情况下是固定的，如 `Resize` 会固定地读取输入字典中的 `"img"` 等字段。我们可以在对应类的文档中了解对输入输出字段的约定。

```{note}
默认情况下，在需要图像尺寸作为**初始化参数**的数据变换 (如Resize, Pad) 中，图像尺寸的顺序均为 (width, height)。在数据变换**返回的字典**中，图像相关的尺寸， 如 `img_shape`、`ori_shape`、`pad_shape` 等，均为 (height, width)。
```

MMCV 为所有的数据变换类提供了一个统一的基类 (`BaseTransform`)：

```python
class BaseTransform(metaclass=ABCMeta):

    def __call__(self, results: dict) -> dict:

        return self.transform(results)

    @abstractmethod
    def transform(self, results: dict) -> dict:
        pass
```

所有的数据变换类都需要继承 `BaseTransform`，并实现 `transform` 方法。`transform` 方法的输入和输出均为一个字典。在**自定义数据变换类**一节中，我们会更详细地介绍如何实现一个数据变换类。

## 数据流水线

如上所述，所有数据变换的输入和输出都是一个字典，而且根据 OpenMMLab 中 [有关数据集的约定](TODO)，数据集中每个样本的基本信息都是一个字典。这样一来，我们可以将所有的数据变换操作首尾相接，组合成为一条数据流水线（data pipeline），输入数据集中样本的信息字典，输出完成一系列处理后的信息字典。

以分类任务为例，我们在下图展示了一个典型的数据流水线。对每个样本，数据集中保存的基本信息是一个如图中最左侧所示的字典，之后每经过一个由蓝色块代表的数据变换操作，数据字典中都会加入新的字段（标记为绿色）或更新现有的字段（标记为橙色）。

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/154197953-bf0b1a16-3f41-4bc7-9e67-b2b9b323d895.png" width="90%"/>
</div>

在配置文件中，数据流水线是一个若干数据变换配置字典组成的列表，每个数据集都需要设置参数 `pipeline` 来定义该数据集需要进行的数据准备操作。如上数据流水线在配置文件中的配置如下：

```python
pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=256, keep_ratio=True),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
    dict(type='ClsFormatBundle')
]

dataset = dict(
    ...
    pipeline=pipeline,
    ...
)
```

## 常用的数据变换类

按照功能，常用的数据变换类可以大致分为数据加载、数据预处理与增强、数据格式化。在 MMCV 中，我们提供了一些常用的数据变换类如下：

### 数据加载

为了支持大规模数据集的加载，通常在 `Dataset` 初始化时不加载数据，只加载相应的路径。因此需要在数据流水线中进行具体数据的加载。

|            class            |                   功能                    |
| :-------------------------: | :---------------------------------------: |
| [`LoadImageFromFile`](TODO) |             根据路径加载图像              |
|  [`LoadAnnotations`](TODO)  | 加载和组织标注信息，如 bbox、语义分割图等 |

### 数据预处理及增强

数据预处理和增强通常是对图像本身进行变换，如裁剪、填充、缩放等。

|              class               |                功能                |
| :------------------------------: | :--------------------------------: |
|          [`Pad`](TODO)           |            填充图像边缘            |
|       [`CenterCrop`](TODO)       |              居中裁剪              |
|       [`Normalize`](TODO)        |          对图像进行归一化          |
|         [`Resize`](TODO)         |     按照指定尺寸或比例缩放图像     |
|      [`RandomResize`](TODO)      |    缩放图像至指定范围的随机尺寸    |
| [`RandomMultiscaleResize`](TODO) | 缩放图像至多个尺寸中的随机一个尺寸 |
|    [`RandomGrayscale`](TODO)     |             随机灰度化             |
|       [`RandomFlip`](TODO)       |            图像随机翻转            |
|   [`MultiScaleFlipAug`](TODO)    |   支持缩放和翻转的测试时数据增强   |

### 数据格式化

数据格式化操作通常是对数据进行的类型转换。

|          class          |               功能                |
| :---------------------: | :-------------------------------: |
|   [`ToTensor`](TODO)    | 将指定的数据转换为 `torch.Tensor` |
| [`ImageToTensor`](TODO) |    将图像转换为 `torch.Tensor`    |

## 自定义数据变换类

要实现一个新的数据变换类，需要继承 `BaseTransform`，并实现 `transform` 方法。这里，我们使用一个简单的翻转变换（`MyFlip`）作为示例：

```python
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyFlip(BaseTransform):
    def __init__(self, direction: str):
        super().__init__()
        self.direction = direction

    def transform(self, results: dict) -> dict:
        img = results['img']
        results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

从而，我们可以实例化一个 `MyFlip` 对象，并将之作为一个可调用对象，来处理我们的数据字典。

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

又或者，在配置文件的 pipeline 中使用 `MyFlip` 变换

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

需要注意的是，如需在配置文件中使用，需要保证 `MyFlip` 类所在的文件在运行时能够被导入。

## 变换包装

变换包装是一种特殊的数据变换类，他们本身并不操作数据字典中的图像、标签等信息，而是对其中定义的数据变换的行为进行增强。

### 字段映射（KeyMapper）

字段映射包装（`KeyMapper`）用于对数据字典中的字段进行映射。例如，一般的图像处理变换都从数据字典中的 `"img"` 字段获得值。但有些时候，我们希望这些变换处理数据字典中其他字段中的图像，比如 `"gt_img"` 字段。

如果配合注册器和配置文件使用的话，在配置文件中数据集的 `pipeline` 中如下例使用字段映射包装：

```python
pipeline = [
    ...
    dict(type='KeyMapper',
        mapping={
            'img': 'gt_img',  # 将 "gt_img" 字段映射至 "img" 字段
            'mask': ...,  # 不使用原始数据中的 "mask" 字段。即对于被包装的数据变换，数据中不包含 "mask" 字段
        },
        auto_remap=True,  # 在完成变换后，将 "img" 重映射回 "gt_img" 字段
        transforms=[
            # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
            dict(type='RandomFlip'),
        ])
    ...
]
```

利用字段映射包装，我们在实现数据变换类时，不需要考虑在 `transform` 方法中考虑各种可能的输入字段名，只需要处理默认的字段即可。

### 随机选择（RandomChoice）和随机执行（RandomApply）

随机选择包装（`RandomChoice`）用于从一系列数据变换组合中随机应用一个数据变换组合。利用这一包装，我们可以简单地实现一些数据增强功能，比如 AutoAugment。

如果配合注册器和配置文件使用的话，在配置文件中数据集的 `pipeline` 中如下例使用随机选择包装：

```python
pipeline = [
    ...
    dict(type='RandomChoice',
        transforms=[
            [
                dict(type='Posterize', bits=4),
                dict(type='Rotate', angle=30.)
            ],  # 第一种随机变化组合
            [
                dict(type='Equalize'),
                dict(type='Rotate', angle=30)
            ],  # 第二种随机变换组合
        ],
        prob=[0.4, 0.6]  # 两种随机变换组合各自的选用概率
        )
    ...
]
```

随机执行包装（`RandomApply`）用于以指定概率随机执行数据变换组合。例如：

```python
pipeline = [
    ...
    dict(type='RandomApply',
        transforms=[dict(type='Rotate', angle=30.)],
        prob=0.3)  # 以 0.3 的概率执行被包装的数据变换
    ...
]
```

### 多目标扩展（TransformBroadcaster）

通常，一个数据变换类只会从一个固定的字段读取操作目标。虽然我们也可以使用 `KeyMapper` 来改变读取的字段，但无法将变换一次性应用于多个字段的数据。为了实现这一功能，我们需要借助多目标扩展包装（`TransformBroadcaster`）。

多目标扩展包装（`TransformBroadcaster`）有两个用法，一是将数据变换作用于指定的多个字段，二是将数据变换作用于某个字段下的一组目标中。

1. 应用于多个字段

   假设我们需要将数据变换应用于 `"lq"` (low-quality) 和 `"gt"` (ground-truth) 两个字段中的图像上。

   ```python
   pipeline = [
       dict(type='TransformBroadcaster',
           # 分别应用于 "lq" 和 "gt" 两个字段，并将二者应设置 "img" 字段
           mapping={'img': ['lq', 'gt']},
           # 在完成变换后，将 "img" 字段重映射回原先的字段
           auto_remap=True,
           # 是否在对各目标的变换中共享随机变量
           # 更多介绍参加后续章节（随机变量共享）
           share_random_params=True,
           transforms=[
               # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
               dict(type='RandomFlip'),
           ])
   ]
   ```

   在多目标扩展的 `mapping` 设置中，我们同样可以使用 `...` 来忽略指定的原始字段。如以下例子中，被包裹的 `RandomCrop` 会对字段 `"img"` 中的图像进行裁剪，并且在字段 `"img_shape"` 存在时更新剪裁后的图像大小。如果我们希望同时对两个图像字段 `"lq"` 和 `"gt"` 进行相同的随机裁剪，但只更新一次 `"img_shape"` 字段，可以通过例子中的方式实现：

   ```python
   pipeline = [
       dict(type='TransformBroadcaster',
           mapping={
               'img': ['lq', 'gt'],
               'img_shape': ['img_shape', ...],
            },
           # 在完成变换后，将 "img" 和 "img_shape" 字段重映射回原先的字段
           auto_remap=True,
           # 是否在对各目标的变换中共享随机变量
           # 更多介绍参加后续章节（随机变量共享）
           share_random_params=True,
           transforms=[
               # `RandomCrop` 类中会操作 "img" 和 "img_shape" 字段。若 "img_shape" 空缺，
               # 则只操作 "img"
               dict(type='RandomCrop'),
           ])
   ]
   ```

2. 应用于一个字段的一组目标

   假设我们需要将数据变换应用于 `"images"` 字段，该字段为一个图像组成的 list。

   ```python
   pipeline = [
       dict(type='TransformBroadcaster',
           # 将 "images" 字段下的每张图片映射至 "img" 字段
           mapping={'img': 'images'},
           # 在完成变换后，将 "img" 字段下的图片重映射回 "images" 字段的列表中
           auto_remap=True,
           # 是否在对各目标的变换中共享随机变量
           share_random_params=True,
           transforms=[
               # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
               dict(type='RandomFlip'),
           ])
   ]
   ```

#### 装饰器 `cache_randomness`

在 `TransformBroadcaster` 中，我们提供了 `share_random_params` 选项来支持在多次数据变换中共享随机状态。例如，在超分辨率任务中，我们希望将随机变换**同步**作用于低分辨率图像和原始图像。如果我们希望在自定义的数据变换类中使用这一功能，需要在类中标注哪些随机变量是支持共享的。这可以通过装饰器 `cache_randomness` 来实现。

以上文中的 `MyFlip` 为例，我们希望以一定的概率随机执行翻转：

```python
from mmcv.transforms.utils import cache_randomness

@TRANSFORMS.register_module()
class MyRandomFlip(BaseTransform):
    def __init__(self, prob: float, direction: str):
        super().__init__()
        self.prob = prob
        self.direction = direction

    @cache_randomness  # 标注该方法的输出为可共享的随机变量
    def do_flip(self):
        flip = True if random.random() > self.prob else False
        return flip

    def transform(self, results: dict) -> dict:
        img = results['img']
        if self.do_flip():
            results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

在上面的例子中，我们用`cache_randomness` 装饰 `do_flip`方法，即将该方法返回值 `flip` 标注为一个支持共享的随机变量。进而，在 `TransformBroadcaster` 对多个目标的变换中，这一变量的值都会保持一致。

#### 装饰器 `avoid_cache_randomness`

在一些情况下，我们无法将数据变换中产生随机变量的过程单独放在类方法中。例如数据变换中使用的来自第三方库的模块，这些模块将随机变量相关的部分封装在了内部，导致无法将其抽出为数据变换的类方法。这样的数据变换无法通过装饰器 `cache_randomness` 标注支持共享的随机变量，进而无法在多目标扩展时共享随机变量。

为了避免在多目标扩展中误用此类数据变换，我们提供了另一个装饰器 `avoid_cache_randomness`，用来对此类数据变换进行标记：

```python
from mmcv.transforms.utils import avoid_cache_randomness

@TRANSFORMS.register_module()
@avoid_cache_randomness
class MyRandomTransform(BaseTransform):

    def transform(self, results: dict) -> dict:
        ...
```

用 `avoid_cache_randomness` 标记的数据变换类，当其实例被 `TransformBroadcaster` 包装且将参数 `share_random_params` 设置为 True 时，会抛出异常，以此提醒用户不能这样使用。

在使用 `avoid_cache_randomness` 时需要注意以下几点：

1. `avoid_cache_randomness` 只用于装饰数据变换类（BaseTransfrom 的子类），而不能用与装饰其他一般的类、类方法或函数
2. 被 `avoid_cache_randomness` 修饰的数据变换作为基类时，其子类将**不会继承**这一特性。如果子类仍无法共享随机变量，则应再次使用 `avoid_cache_randomness` 修饰
3. 只有当一个数据变换具有随机性，且无法共享随机参数时，才需要以 `avoid_cache_randomness` 修饰。无随机性的数据变换不需要修饰
