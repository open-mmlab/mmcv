# Data Transformation

In the OpenMMLab algorithm library, dataset construction and data preparation are decoupled. Usually, the construction of the dataset only parses the dataset and records the basic information of each sample, while the data preparation is a series of data transformations including data loading, preprocessing, formatting, and other operations performed according to the basic information of the sample.

## Design of data transformation

In MMCV, we use various callable data transformation classes to manipulate data. These data transformation classes can accept several configuration parameters for the instantiation and then process the input data dictionary by `__call__` method. All data transformation methods accept a dictionary as the input and produce the output as a dictionary as well. A simple example is as follows:

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

The data transformation class reads some fields of the input dictionary and may add or update some fields. The keys of these fields are mostly fixed. For example, `Resize` will always read fields such as `"img"` in the input dictionary. More information about the conventions for input and output fields could be found in the documentation of the corresponding class.

```{note}
By convention, the order of image shape which is used as **initialization parameters** in data transformation (such as Resize, Pad) is (width, height). In the dictionary returned by the data transformation, the image related shape, such as `img_shape`, `ori_shape`, `pad_shape`, etc., is (height, width).
```

MMCV provides a unified base class called `BaseTransform` for all data transformation classes:

```python
class BaseTransform(metaclass=ABCMeta):

    def __call__(self, results: dict) -> dict:

        return self.transform(results)

    @abstractmethod
    def transform(self, results: dict) -> dict:
        pass
```

All data transformation classes must inherit `BaseTransform` and implement the `transform` method. Both the input and output of the `transform` method are a dictionary. In the **Custom data transformation class** section, we will describe how to implement a data transformation class in more detail.

## Data pipeline

As mentioned above, the inputs and outputs of all data transformations are dictionaries. Moreover, according to the \[Convention on Datasets\] (TODO) in OpenMMLab, the basic information of each sample in the dataset is also a dictionary. This way, we can connect all data transformation operations end to end and combine them into a data pipeline. This pipeline inputs the information dictionary of the samples in the dataset and outputs the information dictionary after a series of processing.

Taking the classification task as an example, we show a typical data pipeline in the figure below. For each sample, the information stored in the dataset is a dictionary, as shown on the far left in the figure. After each data transformation operation represented by the blue block, a new field (marked in green) will be added to the data dictionary or an existing field (marked in orange) will be updated.

<div align=center>
<img src="https://user-images.githubusercontent.com/26739999/154197953-bf0b1a16-3f41-4bc7-9e67-b2b9b323d895.png" width="90%"/>
</div>

The data pipeline is a list of several data transformation configuration dictionaries in the configuration file. Each dataset needs to set the parameter `pipeline` to define the data preparation operations the dataset needs to perform. The configuration of the above data pipeline in the configuration file is as follows:

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

## Common data transformation classes

The commonly used data transformation classes can be roughly divided into data loading, data preprocessing and augmentation, and data formatting. In MMCV, we provide some commonly used classes as follows:

### Data loading

To support the loading of large-scale datasets, data is usually not loaded when `Dataset` is initialized. Only the corresponding path is loaded. Therefore, it is necessary to load specific data in the data pipeline.

|            Class            |                    Feature                     |
| :-------------------------: | :--------------------------------------------: |
| [`LoadImageFromFile`](TODO) |              Load from file path               |
|  [`LoadAnnotations`](TODO)  | Load and organize the annotations (bbox, etc.) |

### Data preprocessing and enhancement

Data preprocessing and augmentation usually involve transforming the image itself, such as cropping, padding, scaling, etc.

|              Class               |                        Feature                         |
| :------------------------------: | :----------------------------------------------------: |
|          [`Pad`](TODO)           |                        Padding                         |
|       [`CenterCrop`](TODO)       |                      Center crop                       |
|       [`Normalize`](TODO)        |                  Image normalization                   |
|         [`Resize`](TODO)         |         Resize to the specified size or ratio          |
|      [`RandomResize`](TODO)      |  Scale the image randomly within the specified range   |
| [`RandomMultiscaleResize`](TODO) | Scale the image to a random size from multiple options |
|    [`RandomGrayscale`](TODO)     |                    Random grayscale                    |
|       [`RandomFlip`](TODO)       |                      Random flip                       |
|   [`MultiScaleFlipAug`](TODO)    |    Support scaling and flipping during the testing     |

### Data formatting

Data formatting operations are type conversions performed on the data.

|          Class          |                   Feature                    |
| :---------------------: | :------------------------------------------: |
|   [`ToTensor`](TODO)    | Convert the specified data to `torch.Tensor` |
| [`ImageToTensor`](TODO) |     Convert the image to `torch.Tensor`      |

## Customize data transformation classes

To implement a new data transformation class, you must inherit `BaseTransform` and implement the `transform` method. Here, we use a simple flip transform (`MyFlip`) as an example:

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

Now, we can instantiate `MyFlip` as a callable object to handle our data dictionary.

```python
import numpy as np

transform = MyFlip(direction='horizontal')
data_dict = {'img': np.random.rand(224, 224, 3)}
data_dict = transform(data_dict)
processed_img = data_dict['img']
```

Alternatively, use `MyFlip` transform in the `pipeline` of the config file.

```python
pipeline = [
    ...
    dict(type='MyFlip', direction='horizontal'),
    ...
]
```

It should be noted that if you want to use it in the configuration file, you must ensure that the file where the `MyFlip` class is located can be imported at the runtime.

## Transform wrapper

Transform wrappers are a special class of data transformations. They do not operate on images, labels or other information in the data dictionary by themselves. Instead, they enhance the behavior of data transformations defined in them.

### KeyMapper

`KeyMapper` is used to map fields in the data dictionary. For example, image processing transforms usually get their values from the `"img"` field in the data dictionary. But sometimes we want these transforms to handle images in other fields in the data dictionary, such as the `"gt_img"` field.

When used with registry and configuration file, the field map wrapper should be used as follows:

```python
pipeline = [
    ...
    dict(type='KeyMapper',
        mapping={
            'img': 'gt_img',  # map "gt_img" to "img"
            'mask': ...,  # The "mask" field in the raw data is not used. That is, for wrapped data transformations, the "mask" field is not included in the data
        },
        auto_remap=True,  # remap "img" back to "gt_img" after the transformation
        transforms=[
            # only need to specify "img" in `RandomFlip`
            dict(type='RandomFlip'),
        ])
    ...
]
```

With `KeyMapper`, we don't need to consider various possible input field names in the `transform` method when we implement the data transformation class. We only need to deal with the default fields.

### RandomChoice and RandomApply

`RandomChoice` is used to randomly select a data transformation pipeline from the given choices. With this wrapper, we can easily implement some data augmentation functions, such as AutoAugment.

In configuration file, you can use `RandomChoice` as follows:

```python
pipeline = [
    ...
    dict(type='RandomChoice',
        transforms=[
            [
                dict(type='Posterize', bits=4),
                dict(type='Rotate', angle=30.)
            ],  # the first combo option
            [
                dict(type='Equalize'),
                dict(type='Rotate', angle=30)
            ],  # the second combo option
        ],
        prob=[0.4, 0.6]  # the prob of each combo
        )
    ...
]
```

`RandomApply` is used to randomly perform a combination of data transformations with a specified probability. For example:

```python
pipeline = [
    ...
    dict(type='RandomApply',
        transforms=[dict(type='Rotate', angle=30.)],
        prob=0.3)  # perform the transformation with prob as 0.3
    ...
]
```

### TransformBroadcaster

Usually, a data transformation class only reads the target of an operation from one field. While we can also use `KeyMapper` to change the fields read, there is no way to apply transformations to the data of multiple fields at once. To achieve this, we need to use the multi-target extension wrapper `TransformBroadcaster`.

`TransformBroadcaster` has two uses, one is to apply data transformation to multiple specified fields, and the other is to apply data transformation to a group of targets under a field.

1. Apply to multiple fields

   Suppose we need to apply a data transformation to images in two fields `"lq"` (low-quality) and `"gt"` (ground-truth).

   ```python
   pipeline = [
       dict(type='TransformBroadcaster',
           # apply to the "lq" and "gt" fields respectively, and set the "img" field to both
           mapping={'img': ['lq', 'gt']},
           # remap the "img" field back to the original field after the transformation
           auto_remap=True,
           # whether to share random variables in the transformation of each target
           # more introduction will be referred in the following chapters (random variable sharing)
           share_random_params=True,
           transforms=[
               # only need to manipulate the "img" field in the `RandomFlip` class
               dict(type='RandomFlip'),
           ])
   ]
   ```

   In the `mapping` setting of the multi-target extension, we can also use `...` to ignore the specified original field. As shown in the following example, the wrapped `RandomCrop` will crop the image in the field `"img"` and update the size of the cropped image if the field `"img_shape"` exists. If we want to do the same random cropping for both image fields `"lq"` and `"gt"` at the same time but update the `"img_shape"` field only once, we can do it as in the example:

   ```python
   pipeline = [
       dict(type='TransformBroadcaster',
           mapping={
               'img': ['lq', 'gt'],
               'img_shape': ['img_shape', ...],
            },
           # remap the "img" and "img_shape" fields back to their original fields after the transformation
           auto_remap=True,
           # whether to share random variables in the transformation of each target
           # more introduction will be referred in the following chapters (random variable sharing)
           share_random_params=True,
           transforms=[
               # "img" and "img_shape" fields are manipulated in the `RandomCrop` class
               # if "img_shape" is missing, only operate on "img"
               dict(type='RandomCrop'),
           ])
   ]
   ```

2. A set of targets applied to a field

   Suppose we need to apply a data transformation to the `"images"` field, which is a list of images.

   ```python
   pipeline = [
       dict(type='TransformBroadcaster',
           # map each image under the "images" field to the "img" field
           mapping={'img': 'images'},
           # remap the images under the "img" field back to the list in the "images" field after the transformation
           auto_remap=True,
           # whether to share random variables in the transformation of each target
           share_random_params=True,
           transforms=[
               # in the `RandomFlip` transformation class, we only need to manipulate the "img" field
               dict(type='RandomFlip'),
           ])
   ]
   ```

#### Decorator `cache_randomness`

In `TransformBroadcaster`, we provide the `share_random_params` option to support sharing random states across multiple data transformations. For example, in a super-resolution task, we want to apply **the same** random transformations **simultaneously** to the low-resolution image and the original image. If we use this function in a custom data transformation class, we need to mark which random variables support sharing in the class. This can be achieved with the decorator `cache_randomness`.

Taking `MyFlip` from the above example, we want to perform flipping randomly with a certain probability:

```python
from mmcv.transforms.utils import cache_randomness

@TRANSFORMS.register_module()
class MyRandomFlip(BaseTransform):
    def __init__(self, prob: float, direction: str):
        super().__init__()
        self.prob = prob
        self.direction = direction

    @cache_randomness  # label the output of the method as a shareable random variable
    def do_flip(self):
        flip = True if random.random() > self.prob else False
        return flip

    def transform(self, results: dict) -> dict:
        img = results['img']
        if self.do_flip():
            results['img'] = mmcv.imflip(img, direction=self.direction)
        return results
```

In the above example, we decorate the `do_flip` method with `cache_randomness`, marking the method return value `flip` as a random variable that supports sharing. Therefore, in the transformation of `TransformBroadcaster` to multiple targets, the value of this variable will remain the same.

#### Decorator `avoid_cache_randomness`

In some cases, we cannot separate the process of generating random variables in data transformation into a class method. For example, modules from third-party libraries used in data transformation encapsulate the relevant parts of random variables inside, making them impossible to be extracted as class methods for data transformation. Such data transformations cannot support shared random variables through the decorator `cache_randomness` annotation, and thus cannot share random variables during multi-objective expansion.

To avoid misuse of such data transformations in multi-object extensions, we provide another decorator, `avoid_cache_randomness`, to mark such data transformations:

```python
from mmcv.transforms.utils import avoid_cache_randomness

@TRANSFORMS.register_module()
@avoid_cache_randomness
class MyRandomTransform(BaseTransform):

    def transform(self, results: dict) -> dict:
        ...
```

Data transformation classes marked with `avoid_cache_randomness` will throw an exception when their instance is wrapped by `TransformBroadcaster` and the parameter `share_random_params` is set to True. This reminds the user not to use it in this way.

There are a few things to keep in mind when using `avoid_cache_randomness`:

1. `avoid_cache_randomness` is only used to decorate data transformation classes (subclasses of `BaseTransfrom`) and cannot be used to decorate other general classes, class methods, or functions
2. When a data transformation decorated with `avoid_cache_randomness` is used as a base class, its subclasses **will not inherit** its feature. If the subclass is still unable to share random variables, `avoid_cache_randomness` should be used again.
3. A data transformation needs to be modified with `avoid_cache_randomness` only when a data transformation is random and cannot share its random parameters. Data transformations without randomness require no decoration
