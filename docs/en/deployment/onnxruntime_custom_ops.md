## ONNX Runtime Custom Ops

<!-- TOC -->

- [ONNX Runtime Custom Ops](#onnx-runtime-custom-ops)
  - [SoftNMS](#softnms)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [RoIAlign](#roialign)
    - [Description](#description-1)
    - [Parameters](#parameters-1)
    - [Inputs](#inputs-1)
    - [Outputs](#outputs-1)
    - [Type Constraints](#type-constraints-1)
  - [NMS](#nms)
    - [Description](#description-2)
    - [Parameters](#parameters-2)
    - [Inputs](#inputs-2)
    - [Outputs](#outputs-2)
    - [Type Constraints](#type-constraints-2)
  - [grid_sampler](#grid_sampler)
    - [Description](#description-3)
    - [Parameters](#parameters-3)
    - [Inputs](#inputs-3)
    - [Outputs](#outputs-3)
    - [Type Constraints](#type-constraints-3)
  - [CornerPool](#cornerpool)
    - [Description](#description-4)
    - [Parameters](#parameters-4)
    - [Inputs](#inputs-4)
    - [Outputs](#outputs-4)
    - [Type Constraints](#type-constraints-4)
  - [cummax](#cummax)
    - [Description](#description-5)
    - [Parameters](#parameters-5)
    - [Inputs](#inputs-5)
    - [Outputs](#outputs-5)
    - [Type Constraints](#type-constraints-5)
  - [cummin](#cummin)
    - [Description](#description-6)
    - [Parameters](#parameters-6)
    - [Inputs](#inputs-6)
    - [Outputs](#outputs-6)
    - [Type Constraints](#type-constraints-6)
  - [MMCVModulatedDeformConv2d](#mmcvmodulateddeformconv2d)
    - [Description](#description-7)
    - [Parameters](#parameters-7)
    - [Inputs](#inputs-7)
    - [Outputs](#outputs-7)
    - [Type Constraints](#type-constraints-7)
  - [MMCVDeformConv2d](#mmcvdeformconv2d)
    - [Description](#description-8)
    - [Parameters](#parameters-8)
    - [Inputs](#inputs-8)
    - [Outputs](#outputs-8)
    - [Type Constraints](#type-constraints-8)

<!-- TOC -->

### SoftNMS

#### Description

Perform soft NMS on `boxes` with `scores`. Read [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503) for detail.

#### Parameters

| Type    | Parameter       | Description                                                    |
| ------- | --------------- | -------------------------------------------------------------- |
| `float` | `iou_threshold` | IoU threshold for NMS                                          |
| `float` | `sigma`         | hyperparameter for gaussian method                             |
| `float` | `min_score`     | score filter threshold                                         |
| `int`   | `method`        | method to do the nms, (0: `naive`, 1: `linear`, 2: `gaussian`) |
| `int`   | `offset`        | `boxes` width or height is (x2 - x1 + offset). (0 or 1)        |

#### Inputs

<dl>
<dt><tt>boxes</tt>: T</dt>
<dd>Input boxes. 2-D tensor of shape (N, 4). N is the number of boxes.</dd>
<dt><tt>scores</tt>: T</dt>
<dd>Input scores. 1-D tensor of shape (N, ).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>dets</tt>: T</dt>
<dd>Output boxes and scores. 2-D tensor of shape (num_valid_boxes, 5), [[x1, y1, x2, y2, score], ...]. num_valid_boxes is the number of valid boxes.</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>Output indices. 1-D tensor of shape (num_valid_boxes, ).</dd>
</dl>

#### Type Constraints

- T:tensor(float32)

### RoIAlign

#### Description

Perform RoIAlign on output feature, used in bbox_head of most two-stage detectors.

#### Parameters

| Type    | Parameter        | Description                                                                                                   |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `int`   | `output_height`  | height of output roi                                                                                          |
| `int`   | `output_width`   | width of output roi                                                                                           |
| `float` | `spatial_scale`  | used to scale the input boxes                                                                                 |
| `int`   | `sampling_ratio` | number of input samples to take for each output sample. `0` means to take samples densely for current models. |
| `str`   | `mode`           | pooling mode in each bin. `avg` or `max`                                                                      |
| `int`   | `aligned`        | If `aligned=0`, use the legacy implementation in MMDetection. Else, align the results more perfectly.         |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
<dt><tt>rois</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 5) given as [[batch_index, x1, y1, x2, y2], ...]. The RoIs' coordinates are the coordinate system of input.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>feat</tt>: T</dt>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element feat[r-1] is a pooled feature map corresponding to the r-th RoI RoIs[r-1].<dd>
</dl>

#### Type Constraints

- T:tensor(float32)

### NMS

#### Description

Filter out boxes has high IoU overlap with previously selected boxes.

#### Parameters

| Type    | Parameter       | Description                                                                                                        |
| ------- | --------------- | ------------------------------------------------------------------------------------------------------------------ |
| `float` | `iou_threshold` | The threshold for deciding whether boxes overlap too much with respect to IoU. Value range \[0, 1\]. Default to 0. |
| `int`   | `offset`        | 0 or 1, boxes' width or height is (x2 - x1 + offset).                                                              |

#### Inputs

<dl>
<dt><tt>bboxes</tt>: T</dt>
<dd>Input boxes. 2-D tensor of shape (num_boxes, 4). num_boxes is the number of input boxes.</dd>
<dt><tt>scores</tt>: T</dt>
<dd>Input scores. 1-D tensor of shape (num_boxes, ).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>indices</tt>: tensor(int32, Linear)</dt>
<dd>Selected indices. 1-D tensor of shape (num_valid_boxes, ). num_valid_boxes is the number of valid boxes.</dd>
</dl>

#### Type Constraints

- T:tensor(float32)

### grid_sampler

#### Description

Perform sample from `input` with pixel locations from `grid`.

#### Parameters

| Type  | Parameter            | Description                                                                                                                                                                                                                                                                                     |
| ----- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `int` | `interpolation_mode` | Interpolation mode to calculate output values. (0: `bilinear` , 1: `nearest`)                                                                                                                                                                                                                   |
| `int` | `padding_mode`       | Padding mode for outside grid values. (0: `zeros`, 1: `border`, 2: `reflection`)                                                                                                                                                                                                                |
| `int` | `align_corners`      | If `align_corners=1`, the extrema (`-1` and `1`) are considered as referring to the center points of the input's corner pixels. If `align_corners=0`, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic. |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the numbers of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>grid</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, outH, outW, 2), where outH and outW is the height and width of offset and output. </dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, C, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### CornerPool

#### Description

Perform CornerPool on `input` features. Read [CornerNet -- Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244) for more details.

#### Parameters

| Type  | Parameter | Description                                                      |
| ----- | --------- | ---------------------------------------------------------------- |
| `int` | `mode`    | corner pool mode, (0: `top`, 1: `bottom`, 2: `left`, 3: `right`) |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input features. 4-D tensor of shape (N, C, H, W). N is the batch size.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output the pooled features. 4-D tensor of shape (N, C, H, W).</dd>
</dl>

#### Type Constraints

- T:tensor(float32)

### cummax

#### Description

Returns a tuple (`values`, `indices`) where `values` is the cumulative maximum elements of `input` in the dimension `dim`. And `indices` is the index location of each maximum value found in the dimension `dim`. Read [torch.cummax](https://pytorch.org/docs/stable/generated/torch.cummax.html) for more details.

#### Parameters

| Type  | Parameter | Description                            |
| ----- | --------- | -------------------------------------- |
| `int` | `dim`     | the dimension to do the operation over |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>The input tensor with various shapes. Tensor with empty element is also supported.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output the cumulative maximum elements of `input` in the dimension `dim`, with the same shape and dtype as `input`.</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>Output the index location of each cumulative maximum value found in the dimension `dim`, with the same shape as `input`.</dd>
</dl>

#### Type Constraints

- T:tensor(float32)

### cummin

#### Description

Returns a tuple (`values`, `indices`) where `values` is the cumulative minimum elements of `input` in the dimension `dim`. And `indices` is the index location of each minimum value found in the dimension `dim`. Read [torch.cummin](https://pytorch.org/docs/stable/generated/torch.cummin.html) for more details.

#### Parameters

| Type  | Parameter | Description                            |
| ----- | --------- | -------------------------------------- |
| `int` | `dim`     | the dimension to do the operation over |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>The input tensor with various shapes. Tensor with empty element is also supported.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output the cumulative minimum elements of `input` in the dimension `dim`, with the same shape and dtype as `input`.</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>Output the index location of each cumulative minimum value found in the dimension `dim`, with the same shape as `input`.</dd>
</dl>

#### Type Constraints

- T:tensor(float32)

### MMCVModulatedDeformConv2d

#### Description

Perform Modulated Deformable Convolution on input feature, read [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168?from=timeline) for detail.

#### Parameters

| Type           | Parameter           | Description                                                                           |
| -------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `list of ints` | `stride`            | The stride of the convolving kernel. (sH, sW)                                         |
| `list of ints` | `padding`           | Paddings on both sides of the input. (padH, padW)                                     |
| `list of ints` | `dilation`          | The spacing between kernel elements. (dH, dW)                                         |
| `int`          | `deformable_groups` | Groups of deformable offset.                                                          |
| `int`          | `groups`            | Split input into groups. `input_channel` should be divisible by the number of groups. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the number of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, deformable_group* 2* kH* kW, outH, outW), where kH and kW is the height and width of weight, outH and outW is the height and width of offset and output.</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>Input mask; 4-D tensor of shape (N, deformable_group* kH* kW, outH, outW), where kH and kW is the height and width of weight, outH and outW is the height and width of offset and output.</dd>
<dt><tt>inputs[3]</tt>: T</dt>
<dd>Input weight; 4-D tensor of shape (output_channel, input_channel, kH, kW).</dd>
<dt><tt>inputs[4]</tt>: T, optional</dt>
<dd>Input bias; 1-D tensor of shape (output_channel).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, output_channel, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVDeformConv2d

#### Description

Perform Deformable Convolution on input feature, read [Deformable Convolutional Network](https://arxiv.org/abs/1703.06211) for detail.

#### Parameters

| Type           | Parameter          | Description                                                                                                                       |
| -------------- | ------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| `list of ints` | `stride`           | The stride of the convolving kernel. (sH, sW)                                                                                     |
| `list of ints` | `padding`          | Paddings on both sides of the input. (padH, padW)                                                                                 |
| `list of ints` | `dilation`         | The spacing between kernel elements. (dH, dW)                                                                                     |
| `int`          | `deformable_group` | Groups of deformable offset.                                                                                                      |
| `int`          | `group`            | Split input into groups. `input_channel` should be divisible by the number of groups.                                             |
| `int`          | `im2col_step`      | DeformableConv2d use im2col to compute convolution. im2col_step is used to split input and offset, reduce memory usage of column. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the numbers of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, deformable_group* 2* kH* kW, outH, outW), where kH and kW is the height and width of weight, outH and outW is the height and width of offset and output.</dd>
<dt><tt>inputs[2]</tt>: T</dt>
<dd>Input weight; 4-D tensor of shape (output_channel, input_channel, kH, kW).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, output_channel, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)
