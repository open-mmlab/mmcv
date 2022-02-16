# 数据变换

从数据集读出数据，到将数据投喂给模型之间，通常需要我们对数据做一系列的处理，包括数据加载、格式化和增强等。与 PyTorch 类似，我们使用**数据变换**类来对数据进行各种操作。进而将一系列数据变换组合为数据流水线（data pipeline）。

## 数据流水线

根据 MMEngine 中 [有关 `Dataset` 的约定](TODO)，`Dataset` 返回字典类型的数据。同样的，在数据准备的过程中，我们也约定所有数据变换类都接受一个字典作为输入，同时输出一个字典用于接下来的变换。从而，我们可以将这些数据变换操作首尾相接，组合成为一个完成整个数据准备流程的数据流水线（data pipeline）。

以分类任务为例，我们在下图展示了一个经典的数据处理流程。其中每个蓝色块代表一个数据变换操作。每个操作都可以在数据字典中加入新的字段（标记为绿色）或更新现有的字段（标记为橙色）。

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/154197953-bf0b1a16-3f41-4bc7-9e67-b2b9b323d895.png" width="90%"/>
</div>

常用的数据变换类可以大致分为数据加载、数据预处理、数据增强和数据格式化。还有一类特殊数据变换类，称为变换包装，它们不操作数据，而是为其中包含的一组数据变换提供额外的功能。

## 常用的数据变换类

在 MMCV 中，我们提供了一些在常用的数据变换类如下：

### 数据加载

为了支持大规模数据集的加载，通常在 `Dataset` 初始化时不加载数据，只加载相应的路径。因此需要在数据流水线中进行具体数据的加载。

| class                         | 功能                       |
| :---------------------------: | :------------------------: |
| [`LoadImageFromFile`](TODO)   | 根据路径加载图像           |

### 数据预处理及增强

数据预处理和增强通常是对图像本身进行变换，如裁剪、填充、缩放等。

| class                            | 功能                               |
| :------------------------------: | :--------------------------------: |
| [`Pad`](TODO)                    | 填充图像边缘                       |
| [`CenterCrop`](TODO)             | 居中裁剪                           |
| [`Normalize`](TODO)              | 对图像进行归一化                   |
| [`Resize`](TODO)                 | 按照指定尺寸或比例缩放图像         |
| [`RandomResize`](TODO)           | 缩放图像至指定范围的随机尺寸       |
| [`RandomMultiscaleResize`](TODO) | 缩放图像至多个尺寸中的随机一个尺寸 |
| [`RandomGrayscale`](TODO)        | 随机灰度化                         |
| [`RandomFlip`](TODO)             | 图像随机翻转                       |
| [`MultiScaleFlipAug`](TODO)      | 支持缩放和翻转的测试时数据增强     |

### 数据格式化

数据格式化操作通常是对数据进行的类型转换。

| class                         | 功能                               |
| :---------------------------: | :--------------------------------: |
| [`ToTensor`](TODO)            | 将指定的数据转换为 `torch.Tensor`  |
| [`ImageToTensor`](TODO)       | 将图像转换为 `torch.Tensor`        |


## 变换包装

变换包装是一种特殊的数据变换类，他们本身并不操作数据字典中的图像、标签等信息，而是对其中定义的数据变换的行为进行增强。

### 字段映射（Remap）

字段映射包装（`Remap`）用于对数据字典中的字段进行映射。例如，一般的图像处理变换都从数据字典中的 `"img"` 字段获得值。但有些时候，我们希望这些变换处理数据字典中其他字段中的图像，比如 `"gt_img"` 字段。

如果配合注册器和配置文件使用的话，在配置文件中数据集的 `pipeline` 中如下例使用字段映射包装：

```python
pipeline = [
    ...
    dict(type='Remap',
        input_mapping={'img': 'gt_img'},  # 将 "gt_img" 字段映射至 "img" 字段
        inplace=True,  # 在完成变换后，将 "img" 重映射回 "gt_img" 字段
        transforms=[
            # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
            dict(type='RandomFlip'),
        ])
    ...
]
```

利用字段映射包装，我们在实现数据变换类时，不需要考虑在 `transform` 方法中考虑各种可能的输入字段名，只需要处理默认的字段即可。

### 随机选择（RandomChoice）

随机选择包装（`RandomChoice`）用于从一系列数据变换组合中随机应用一个数据变换组合。利用这一包装，我们可以简单地实现一些数据增强功能，比如 AutoAugment。

如果配合注册器和配置文件使用的话，在配置文件中数据集的 `pipeline` 中如下例使用随机选择包装：

```python
pipeline = [
    ...
    dict(type='RandomChoice',
        pipelines=[
            [
                dict(type='Posterize', bits=4),
                dict(type='Rotate', angle=30.)
            ],  # 第一种随机变化组合
            [
                dict(type='Equalize'),
                dict(type='Rotate', angle=30)
            ],  # 第二种随机变换组合
        ],
        pipeline_probs=[0.4, 0.6]  # 两种随机变换组合各自的选用概率
        )
    ...
]
```

### 多目标扩展（ApplyToMultiple）

通常，一个数据变换类只会从一个固定的字段读取操作目标。虽然我们也可以使用 `Remap` 来改变读取的字段，但无法将变换一次性应用于多个字段的数据。为了实现这一功能，我们需要借助多目标扩展包装（`ApplyToMultiple`）。

多目标扩展包装（`ApplyToMultiple`）有两个用法，一是将数据变换作用于指定的多个字段，二是将数据变换作用于某个字段下的一组目标中。

1. 应用于多个字段

   假设我们需要将数据变换应用于 `"lq"` (low-quanlity) 和 `"gt"` (ground-truth) 两个字段中的图像上。

   ```python
   pipeline = [
       dict(type='ApplyToMultiple',
           # 分别应用于 "lq" 和 "gt" 两个字段，并将二者应设置 "img" 字段
           input_mapping={'img': ['lq', 'gt']},
           # 在完成变换后，将 "img" 字段重映射回原先的字段
           inplace=True,
           # 是否在对各目标的变换中共享随机变量
           # 更多介绍参加后续章节（随机变量共享）
           share_random_param=True,
           transforms=[
               # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
               dict(type='RandomFlip'),
           ])
   ]
   ```

2. 应用于一个字段的一组目标

   假设我们需要将数据变换应用于 `"images"` 字段中一个 list 的图像。

   ```python
   pipeline = [
       dict(type='ApplyToMultiple',
           # 将 "images" 字段下的每张图片映射至 "img" 字段
           input_mapping={'img': 'images'},
           # 在完成变换后，将 "img" 字段下的图片重映射回 "images" 字段的列表中
           inplace=True,
           # 是否在对各目标的变换中共享随机变量
           # 更多介绍参加后续章节（随机变量共享）
           share_random_param=True,
           transforms=[
               # 在 `RandomFlip` 变换类中，我们只需要操作 "img" 字段即可
               dict(type='RandomFlip'),
           ])
   ]
   ```

## 自定义数据变换类

在 MMCV 中，我们提供了统一的数据变换的基类 `BaseTransform`。这是一个抽象类，定义了数据变换类的接口。

如果要自定义一个新的数据变换类，只需要继承 `BaseTransform`，并实现 `transform` 函数即可。这里，我们使用一个简单的随机翻转变换（`MyRandomFlip`）作为示例：

```python
import random
import mmcv
from mmcv.transforms import BaseTransform, TRANSFORMS

@TRANSFORMS.register_module()
class MyRandomFlip(BaseTransform):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    def transform(self, results: dict) -> dict:
        img = results['img']
        flip = True if random.random() > self.prob else False
        if flip:
            results['img'] = mmcv.imflip(img)
        return results
```

进而，我们可以实例化一个 `MyRandomFlip` 对象，并将之作为一个可调用对象，来处理我们的数据字典。

```python
import numpy as np
transform = MyRandomFlip(prob=1.0)
data_info = dict(img=np.random.rand(224, 224, 3))
data_info = transform(data_info)
processed_img = data_info['img']
```

又或者，在配置文件的 pipeline 字段中添加 `MyRandomFlip` 变换

```python
pipeline = [
    ...
    dict(type='MyRandomFlip', prob=0.5),
    ...
]
```

需要注意的是，如需在配置文件中使用，需要保证 `MyRandomFlip` 类所在的文件在运行时能够被导入。

### 随机变量共享

有些情况下，我们希望在多次数据变换中共享随机状态。例如，在超分辨率任务中，我们希望将随机变换**同步**作用于低分辨率图像和原始图像。

在 `ApplyToMultiple` 中，我们提供了 `share_random_param` 选项来启用这一功能。而为了使这一功能生效，我们需要在数据变换类中标注哪些随机变量是支持共享的。

以上文中的 `MyRandomFlip` 为例：

```python
from mmcv.transforms.utils import cacheable_method

@TRANSFORMS.register_module()
class MyRandomFlip(BaseTransform):
    def __init__(self, prob: float):
        super().__init__()
        self.prob = prob

    @cacheable_method  # 标注该方法的输出为可共享的随机变量
    def do_flip(self):
        flip = True if random.random() > self.prob else False
        return flip

    def transform(self, results: dict) -> dict:
        img = results['img']
        if self.do_flip():
            results['img'] = mmcv.imflip(img)
        return results
```

通过 `cacheable_method` 装饰器，方法返回值 `flip` 被标注为一个支持共享的随机变量。进而，在 `ApplyToMultiple` 对多个目标的变换中，这一变量的值都会保持一致。

如果你对 `ApplyToMultiple` 是如何开关这一功能感到好奇，深入了解我们会发现，它使用了上下文管理器 `cache_random_params` 来在特定范围内启用数据变换的随机变量共享。我们可以通过一个小例子来体验这一功能。

```python
>>> import random
>>> from mmcv.transforms import BaseTransform
>>> from mmcv.transforms.utils import cacheable_method, cache_random_params
>>>
>>> class RandomNumber(BaseTransform):
...     @cacheable_method  # 标注该方法的输出为可共享的随机变量
...     def get_cached_random(self):
...         return random.random()
...
...     def transform(self, results: dict) -> dict:
...         results['cache'] = self.get_cached_random()
...         results['no_cache'] = random.random()
...         return results
>>>
>>> transform = RandomNumber()
>>> # 没有 `cache_random_params` 时，被标注的随机变量也不会在多次调用中共享
>>> for i in range(3):
...     data_dict = transform({})
...     print(f'{data_dict["cache"]:.4f}, {data_dict["no_cache"]:.4f}')
0.7994, 0.1712
0.5317, 0.5089
0.6758, 0.0542
>>> # 在 `cache_random_params` 中，只有被标注的随机变量会在多次调用中共享
>>> with cache_random_params(transform):
...     for i in range(3):
...         data_dict = transform({})
...         print(f'{data_dict["cache"]:.4f}, {data_dict["no_cache"]:.4f}')
0.9899, 0.5399
0.9899, 0.4246
0.9899, 0.9384
```
