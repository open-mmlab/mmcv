## ONNX Runtime自定义算子

<!-- TOC -->

- [ONNX Runtime自定义算子](#onnx-runtime自定义算子)
  - [SoftNMS](#softnms)
    - [描述](#描述)
    - [模型参数](#模型参数)
    - [输入](#输入)
    - [输出](#输出)
    - [类型约束](#类型约束)
  - [RoIAlign](#roialign)
    - [描述](#描述-1)
    - [模型参数](#模型参数-1)
    - [输入](#输入-1)
    - [输出](#输出-1)
    - [类型约束](#类型约束-1)
  - [NMS](#nms)
    - [描述](#描述-2)
    - [模型参数](#模型参数-2)
    - [输入](#输入-2)
    - [输出](#输出-2)
    - [类型约束](#类型约束-2)
  - [grid_sampler](#grid_sampler)
    - [描述](#描述-3)
    - [模型参数](#模型参数-3)
    - [输入](#输入-3)
    - [输出](#输出-3)
    - [类型约束](#类型约束-3)
  - [CornerPool](#cornerpool)
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
  - [MMCVModulatedDeformConv2d](#mmcvmodulateddeformconv2d)
    - [描述](#描述-7)
    - [模型参数](#模型参数-7)
    - [输入](#输入-7)
    - [输出](#输出-7)
    - [类型约束](#类型约束-7)

<!-- TOC -->

### SoftNMS

#### 描述

根据`scores`计算`boxes`的soft NMS。 请阅读[Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503)了解细节。

#### 模型参数

| 类型    | 参数名          | 描述                                                    |
| ------- | --------------- | ------------------------------------------------------- |
| `float` | `iou_threshold` | 用来判断候选框重合度的阈值，取值范围\[0, 1\]。默认值为0 |
| `float` | `sigma`         | 高斯方法的超参数                                        |
| `float` | `min_score`     | NMS的score阈值                                          |
| `int`   | `method`        | NMS的计算方式, (0: `naive`, 1: `linear`, 2: `gaussian`) |
| `int`   | `offset`        | 用来计算候选框的宽高(x2 - x1 + offset)。可选值0或1      |

#### 输入

<dl>
<dt><tt>boxes</tt>: T</dt>
<dd>输入候选框。形状为(N, 4)的二维张量，N为候选框数量。</dd>
<dt><tt>scores</tt>: T</dt>
<dd>输入得分。形状为(N, )的一维张量。</dd>
</dl>

#### 输出

<dl>
<dt><tt>dets</tt>: T</dt>
<dd>输出的检测框与得分。形状为(num_valid_boxes, 5)的二维张量，内容为[[x1, y1, x2, y2, score], ...]。num_valid_boxes是合法的检测框数量。</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>输出序号。形状为(num_valid_boxes, )的一维张量。</dd>
</dl>

#### 类型约束

- T:tensor(float32)

### RoIAlign

#### 描述

在特征图上计算RoIAlign，通常在双阶段目标检测模型的bbox_head中使用

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
<dt><tt>input</tt>: T</dt>
<dd>输入特征图；形状为(N, C, H, W)的四维张量，其中N为batch大小，C为输入通道数，H和W为输入特征图的高和宽。</dd>
<dt><tt>rois</tt>: T</dt>
<dd>需要进行池化的感兴趣区域；形状为(num_rois, 5)的二维张量，内容为[[batch_index, x1, y1, x2, y2], ...]。rois的坐标为输入特征图的坐标系。</dd>
</dl>

#### 输出

<dl>
<dt><tt>feat</tt>: T</dt>
<dd>池化的输出；形状为(num_rois, C, output_height, output_width)的四维张量。每个输出特征feat[i]都与输入感兴趣区域rois[i]一一对应。<dd>
</dl>

#### 类型约束

- T:tensor(float32)

### NMS

#### 描述

根据IoU阈值对候选框进行非极大值抑制。

#### 模型参数

| 类型    | 参数名          | 描述                                                    |
| ------- | --------------- | ------------------------------------------------------- |
| `float` | `iou_threshold` | 用来判断候选框重合度的阈值，取值范围\[0, 1\]。默认值为0 |
| `int`   | `offset`        | 用来计算候选框的宽高(x2 - x1 + offset)。可选值0或1      |

#### 输入

<dl>
<dt><tt>boxes</tt>: T</dt>
<dd>输入候选框。形状为(N, 4)的二维张量，N为候选框数量。</dd>
<dt><tt>scores</tt>: T</dt>
<dd>输入得分。形状为(N, )的一维张量。</dd>
</dl>

#### 输出

<dl>
<dt><tt>indices</tt>: tensor(int32, Linear)</dt>
<dd>被选中的候选框索引。形状为(num_valid_boxes, )的一维张量，num_valid_boxes表示被选上的候选框数量。</dd>
</dl>

#### 类型约束

- T:tensor(float32)

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
<dt><tt>input</tt>: T</dt>
<dd>输入特征；形状为(N, C, inH, inW)的四维张量，其中N为batch大小，C为输入通道数，inH和inW为输入特征图的高和宽。</dd>
<dt><tt>grid</tt>: T</dt>
<dd>输入网格；形状为(N, outH, outW, 2)的四维张量，outH和outW为输出的高和宽。 </dd>
</dl>

#### 输出

<dl>
<dt><tt>output</tt>: T</dt>
<dd>输出特征；形状为(N, C, outH, outW)的四维张量。</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)

### CornerPool

#### 描述

对`input`计算CornerPool。请阅读[CornerNet -- Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244)了解更多细节。

#### 模型参数

| 类型  | 参数名 | 描述                                                     |
| ----- | ------ | -------------------------------------------------------- |
| `int` | `mode` | 池化模式。(0: `top`, 1: `bottom`, 2: `left`, 3: `right`) |

#### 输入

<dl>
<dt><tt>input</tt>: T</dt>
<dd>输入特征；形状为(N, C, H, W)的四维张量，其中N为batch大小，C为输入通道数，H和W为输入特征图的高和宽。</dd>
</dl>

#### 输出

<dl>
<dt><tt>output</tt>: T</dt>
<dd>输出特征；形状为(N, C, H, W)的四维张量。</dd>
</dl>

#### 类型约束

- T:tensor(float32)

### cummax

#### 描述

返回一个元组(`values`, `indices`)，其中`values`为`input`第`dim`维的累计最大值，`indices`为第`dim`维最大值位置。请阅读[torch.cummax](https://pytorch.org/docs/stable/generated/torch.cummax.html)了解更多细节。

#### 模型参数

| 类型  | 参数名 | 描述               |
| ----- | ------ | ------------------ |
| `int` | `dim`  | 进行累计计算的维度 |

#### 输入

<dl>
<dt><tt>input</tt>: T</dt>
<dd>输入张量；可以使任意形状；也支持空Tensor</dd>
</dl>

#### 输出

<dl>
<dt><tt>output</tt>: T</dt>
<dd>`input`第`dim`维的累计最大值，形状与`input`相同。类型和`input`一致</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>第`dim`维最大值位置，形状与`input`相同。</dd>
</dl>

#### 类型约束

- T:tensor(float32)

### cummin

#### 描述

返回一个元组(`values`, `indices`)，其中`values`为`input`第`dim`维的累计最小值，`indices`为第`dim`维最小值位置。请阅读[torch.cummin](https://pytorch.org/docs/stable/generated/torch.cummin.html)了解更多细节。

#### 模型参数

| 类型  | 参数名 | 描述               |
| ----- | ------ | ------------------ |
| `int` | `dim`  | 进行累计计算的维度 |

#### 输入

<dl>
<dt><tt>input</tt>: T</dt>
<dd>输入张量；可以是任意形状；也支持空Tensor</dd>
</dl>

#### 输出

<dl>
<dt><tt>output</tt>: T</dt>
<dd>`input`第`dim`维的累计最小值，形状与`input`相同。类型和`input`一致</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>第`dim`维最小值位置，形状与`input`相同。</dd>
</dl>

#### 类型约束

- T:tensor(float32)

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
<dd>输入特征；形状为(N, C, inH, inW)的四维张量，其中N为batch大小，C为输入通道数，inH和inW为输入特征图的高和宽。</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>输入偏移量；形状为(N, deformable_group* 2* kH* kW, outH, outW)的四维张量，kH和kW为输入特征图的高和宽，outH和outW为输入特征图的高和宽。</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>输入掩码；形状为(N, deformable_group* kH* kW, outH, outW)的四维张量。</dd>
<dt><tt>inputs[3]</tt>: T</dt>
<dd>输入权重；形状为(output_channel, input_channel, kH, kW)的四维张量。</dd>
<dt><tt>inputs[4]</tt>: T, optional</dt>
<dd>输入偏移量；形状为(output_channel)的一维张量。</dd>
</dl>

#### 输出

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>输出特征；形状为(N, output_channel, outH, outW)的四维张量。</dd>
</dl>

#### 类型约束

- T:tensor(float32, Linear)
