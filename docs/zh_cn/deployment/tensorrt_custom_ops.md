## TensorRT自定义算子

<!-- TOC -->

- [TensorRT自定义算子](#tensorrt自定义算子)
  - [MMCVRoIAlign](#mmcvroialign)
    - [描述](#描述)
    - [模型参数](#模型参数)
    - [输入](#输入)
    - [输出](#输出)
    - [类型约束](#类型约束)
  - [ScatterND](#scatternd)
    - [描述](#描述-1)
    - [模型参数](#模型参数-1)
    - [输入](#输入-1)
    - [输出](#输出-1)
    - [类型约束](#类型约束-1)
  - [NonMaxSuppression](#nonmaxsuppression)
    - [描述](#描述-2)
    - [模型参数](#模型参数-2)
    - [输入](#输入-2)
    - [输出](#输出-2)
    - [类型约束](#类型约束-2)
  - [MMCVDeformConv2d](#mmcvdeformconv2d)
    - [描述](#描述-3)
    - [模型参数](#模型参数-3)
    - [输入](#输入-3)
    - [输出](#输出-3)
    - [类型约束](#类型约束-3)
  - [grid_sampler](#grid_sampler)
    - [描述](#描述-4)
    - [模型参数](#模型参数-4)
    - [输入](#输入-4)
    - [输出](#输出-4)
    - [类型约束](#类型约束-4)
  - [cummax](#cummax)
    - [描述](#描述-5)
    - [模型参数](#模型参数-5)
    - [输入](#输入-5)
    - [输出](#输出-5)
    - [类型约束](#类型约束-5)
  - [cummin](#cummin)
    - [描述](#描述-6)
    - [模型参数](#模型参数-6)
    - [输入](#输入-6)
    - [输出](#输出-6)
    - [类型约束](#类型约束-6)
  - [MMCVInstanceNormalization](#mmcvinstancenormalization)
    - [描述](#描述-7)
    - [模型参数](#模型参数-7)
    - [输入](#输入-7)
    - [输出](#输出-7)
    - [类型约束](#类型约束-7)
  - [MMCVModulatedDeformConv2d](#mmcvmodulateddeformconv2d)
    - [描述](#描述-8)
    - [模型参数](#模型参数-8)
    - [输入](#输入-8)
    - [输出](#输出-8)
    - [类型约束](#类型约束-8)

<!-- TOC -->

### MMCVRoIAlign

#### 描述

在特征图上计算RoIAlign，在多数双阶段目标检测模型的bbox_head中使用

#### 模型参数

| 类型    | 参数名           | 描述                                                    |
| ------- | ---------------- | ------------------------------------------------------- |
| `int`   | `output_height`  | roi特征的输出高度                                       |
| `int`   | `output_width`   | roi特征的输出宽度                                       |
| `float` | `spatial_scale`  | 输入检测框的缩放系数                                    |
| `int`   | `sampling_ratio` | 输出的采样率。`0`表示使用密集采样                       |
| `str`   | `mode`           | 池化方式。 `avg`或`max`                                 |
| `int`   | `aligned`        | 如果`aligned=1`，则像素会进行-0.5的偏移以达到更好的对齐 |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入特征图；形状为(N, C, H, W)的四维张量，其中N为batch大小，C为输入通道数，H和W为输入特征图的高和宽。</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>需要进行池化的感兴趣区域；形状为(num_rois, 5)的二维张量，内容为[[batch_index, x1, y1, x2, y2], ...]。rois的坐标为输入特征图的坐标系。</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>池化的输出；形状为(num_rois, C, output_height, output_width)的四维张量。每个输出特征feat[i]都与输入感兴趣区域rois[i]一一对应。<dd>
</dl>
#### 类型约束

- T:tensor(float32, Linear)

### ScatterND

#### 描述

ScatterND接收三个输入，分别为秩为r >= 1的`data`，秩为q >= 1的`indices`以及秩为 q + r - indices.shape\[-1\] -1 的`update`。输出的计算方式为：首先创建一个`data`的拷贝，然后根据`indces`的值使用`update`对拷贝的`data`进行更新。注意`indices`中不应该存在相同的条目，也就是说对同一个位置进行一次以上的更新是不允许的。

输出的计算方式可以参考如下代码：

```python
  output = np.copy(data)
  update_indices = indices.shape[:-1]
  for idx in np.ndindex(update_indices):
      output[indices[idx]] = updates[idx]
```

#### 模型参数

无

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>秩为r >= 1的输入`data`</dd>

<dt><tt>inputs[1]</tt>: tensor(int32, Linear)</dt>
<dd>秩为q >= 1的输入`update`</dd>

<dt><tt>inputs[2]</tt>: T</dt>
<dd>秩为 q + r - indices.shape[-1] -1 的输入`update`</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>秩为r >= 1的输出张量</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear), tensor(int32, Linear)

### NonMaxSuppression

#### 描述

根据IoU阈值对候选框进行非极大值抑制。

#### 模型参数

| 类型    | 参数名                       | 描述                                                                                         |
| ------- | ---------------------------- | -------------------------------------------------------------------------------------------- |
| `int`   | `center_point_box`           | 0 - 候选框的格式为\[y1, x1, y2, x2\]， 1-候选框的格式为\[x_center, y_center, width, height\] |
| `int`   | `max_output_boxes_per_class` | 每一类最大的输出检测框个数。默认为0，输出检测框个数等于输入候选框数                          |
| `float` | `iou_threshold`              | 用来判断候选框重合度的阈值，取值范围\[0, 1\]。默认值为0                                      |
| `float` | `score_threshold`            | 用来判断候选框是否合法的阈值                                                                 |
| `int`   | `offset`                     | 检测框长宽计算方式为(x2 - x1 + offset)，可选值0或1                                           |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入候选框。形状为(num_batches, spatial_dimension, 4)的三维张量</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>输入得分。形状为(num_batches, num_classes, spatial_dimension)的三维张量</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: tensor(int32, Linear)</dt>
<dd>被选中的候选框索引。形状为(num_selected_indices, 3)的二维张量。每一行内容为[batch_index, class_index, box_index]。</dd>
<dd>其中 num_selected_indices=num_batches* num_classes* min(max_output_boxes_per_class, spatial_dimension)。</dd>
<dd>所有未被选中的候选框索引都会被填充为-1</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)

### MMCVDeformConv2d

#### 描述

在输入特征上计算Deformable Convolution，请阅读[Deformable Convolutional Network](https://arxiv.org/abs/1703.06211)了解更多细节。

#### 模型参数

| 类型           | 参数名             | 描述                                                                                          |
| -------------- | ------------------ | --------------------------------------------------------------------------------------------- |
| `list of ints` | `stride`           | 卷积的步长 (sH, sW)                                                                           |
| `list of ints` | `padding`          | 输入特征填充大小 (padH, padW)                                                                 |
| `list of ints` | `dilation`         | 卷积核各元素间隔 (dH, dW)                                                                     |
| `int`          | `deformable_group` | 可变偏移量的分组                                                                              |
| `int`          | `group`            | 卷积分组数，`input_channel`会根据这个值被分为数个分组进行计算                                 |
| `int`          | `im2col_step`      | 可变卷积使用im2col计算卷积。输入与偏移量会以im2col_step为步长分块计算，减少临时空间的使用量。 |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入特征；形状为(N, C, inH, inW)的四维张量，其中N为batch大小，C为输入通道数，inH和inW为输入特征图的高和宽</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>输入偏移量；形状为(N, deformable_group* 2* kH* kW, outH, outW)的四维张量，kH和kW为输入特征图的高和宽，outH和outW为输入特征图的高和宽</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>输入权重；形状为(output_channel, input_channel, kH, kW)的四维张量</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>输出特征；形状为(N, output_channel, outH, outW)的四维张量</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)

### grid_sampler

#### 描述

根据`grid`的像素位置对`input`进行网格采样。

#### 模型参数

| 类型  | 参数名               | 描述                                                                                                                                                 |
| ----- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `int` | `interpolation_mode` | 计算输出使用的插值模式。(0: `bilinear` , 1: `nearest`)                                                                                               |
| `int` | `padding_mode`       | 边缘填充模式。(0: `zeros`, 1: `border`, 2: `reflection`)                                                                                             |
| `int` | `align_corners`      | 如果`align_corners=1`，则极值(`-1`和`1`)会被当做输入边缘像素的中心点。如果`align_corners=0`，则它们会被看做是边缘像素的边缘点,减小分辨率对采样的影响 |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入特征；形状为(N, C, inH, inW)的四维张量，其中N为batch大小，C为输入通道数，inH和inW为输入特征图的高和宽</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>输入网格；形状为(N, outH, outW, 2)的四维张量，outH和outW为输出的高和宽 </dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>输出特征；形状为(N, C, outH, outW)的四维张量</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)

### cummax

#### 描述

返回一个元组(`values`, `indices`)，其中`values`为`input`第`dim`维的累计最大值，`indices`为第`dim`维最大值位置。请阅读[torch.cummax](https://pytorch.org/docs/stable/generated/torch.cummax.html)了解更多细节。

#### 模型参数

| 类型  | 参数名 | 描述               |
| ----- | ------ | ------------------ |
| `int` | `dim`  | 进行累计计算的维度 |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入张量；可以使任意形状</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>`input`第`dim`维的累计最大值，形状与`input`相同。类型和`input`一致</dd>
<dt><tt>outputs[1]</tt>: (int32, Linear)</dt>
<dd>第`dim`维最大值位置，形状与`input`相同</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)

### cummin

#### 描述

返回一个元组(`values`, `indices`)，其中`values`为`input`第`dim`维的累计最小值，`indices`为第`dim`维最小值位置。请阅读[torch.cummin](https://pytorch.org/docs/stable/generated/torch.cummin.html)了解更多细节。

#### 模型参数

| 类型  | 参数名 | 描述               |
| ----- | ------ | ------------------ |
| `int` | `dim`  | 进行累计计算的维度 |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入张量；可以使任意形状</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>`input`第`dim`维的累计最小值，形状与`input`相同。类型和`input`一致</dd>
<dt><tt>outputs[1]</tt>: (int32, Linear)</dt>
<dd>第`dim`维最小值位置，形状与`input`相同</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)

### MMCVInstanceNormalization

#### 描述

对特征计算instance normalization，请阅读[Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)了解更多详细信息。

#### 模型参数

| 类型    | 参数名    | 描述                         |
| ------- | --------- | ---------------------------- |
| `float` | `epsilon` | 用来避免除0错误。默认为1e-05 |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入特征。形状为(N, C, H， W)的四维张量，其中N为batch大小，C为输入通道数，H和W为输入特征图的高和宽</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>输入缩放系数。形状为(C，)的一维张量</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>输入偏移量。形状为(C，)的一维张量</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>输出特征。形状为(N, C, H， W)的四维张量</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)

### MMCVModulatedDeformConv2d

#### 描述

在输入特征上计算Modulated Deformable Convolution，请阅读[Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168?from=timeline)了解更多细节。

#### 模型参数

| 类型           | 参数名              | 描述                                                          |
| -------------- | ------------------- | ------------------------------------------------------------- |
| `list of ints` | `stride`            | 卷积的步长 (sH, sW)                                           |
| `list of ints` | `padding`           | 输入特征填充大小 (padH, padW)                                 |
| `list of ints` | `dilation`          | 卷积核各元素间隔 (dH, dW)                                     |
| `int`          | `deformable_groups` | 可变偏移量的分组，通常置位1即可                               |
| `int`          | `groups`            | 卷积分组数，`input_channel`会根据这个值被分为数个分组进行计算 |

#### 输入

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>输入特征；形状为(N, C, inH, inW)的四维张量，其中N为batch大小，C为输入通道数，inH和inW为输入特征图的高和宽</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>输入偏移量；形状为(N, deformable_group* 2* kH* kW, outH, outW)的四维张量，kH和kW为输入特征图的高和宽，outH和outW为输入特征图的高和宽</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>输入掩码；形状为(N, deformable_group* kH* kW, outH, outW)的四维张量</dd>
<dt><tt>inputs[3]</tt>: T</dt>
<dd>输入权重；形状为(output_channel, input_channel, kH, kW)的四维张量</dd>
<dt><tt>inputs[4]</tt>: T, optional</dt>
<dd>输入偏移量；形状为(output_channel)的一维张量</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>输出特征；形状为(N, output_channel, outH, outW)的四维张量</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)
