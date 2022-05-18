## TensorRT Custom Ops

<!-- TOC -->

- [TensorRT Custom Ops](#tensorrt-custom-ops)
  - [MMCVRoIAlign](#mmcvroialign)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [ScatterND](#scatternd)
    - [Description](#description-1)
    - [Parameters](#parameters-1)
    - [Inputs](#inputs-1)
    - [Outputs](#outputs-1)
    - [Type Constraints](#type-constraints-1)
  - [NonMaxSuppression](#nonmaxsuppression)
    - [Description](#description-2)
    - [Parameters](#parameters-2)
    - [Inputs](#inputs-2)
    - [Outputs](#outputs-2)
    - [Type Constraints](#type-constraints-2)
  - [MMCVDeformConv2d](#mmcvdeformconv2d)
    - [Description](#description-3)
    - [Parameters](#parameters-3)
    - [Inputs](#inputs-3)
    - [Outputs](#outputs-3)
    - [Type Constraints](#type-constraints-3)
  - [grid_sampler](#grid_sampler)
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
  - [MMCVInstanceNormalization](#mmcvinstancenormalization)
    - [Description](#description-7)
    - [Parameters](#parameters-7)
    - [Inputs](#inputs-7)
    - [Outputs](#outputs-7)
    - [Type Constraints](#type-constraints-7)
  - [MMCVModulatedDeformConv2d](#mmcvmodulateddeformconv2d)
    - [Description](#description-8)
    - [Parameters](#parameters-8)
    - [Inputs](#inputs-8)
    - [Outputs](#outputs-8)
    - [Type Constraints](#type-constraints-8)

<!-- TOC -->

### MMCVRoIAlign

#### Description

Perform RoIAlign on output feature, used in bbox_head of most two stage
detectors.

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
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 5) given as [[batch_index, x1, y1, x2, y2], ...]. The RoIs' coordinates are the coordinate system of inputs[0].</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element output[0][r-1] is a pooled feature map corresponding to the r-th RoI inputs[1][r-1].<dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### ScatterND

#### Description

ScatterND takes three inputs `data` tensor of rank r >= 1, `indices` tensor of rank q >= 1, and `updates` tensor of rank q + r - indices.shape\[-1\] - 1. The output of the operation is produced by creating a copy of the input `data`, and then updating its value to values specified by updates at specific index positions specified by `indices`. Its output shape is the same as the shape of `data`. Note that `indices` should not have duplicate entries. That is, two or more updates for the same index-location is not supported.

The `output` is calculated via the following equation:

```python
  output = np.copy(data)
  update_indices = indices.shape[:-1]
  for idx in np.ndindex(update_indices):
      output[indices[idx]] = updates[idx]
```

#### Parameters

None

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Tensor of rank r>=1.</dd>

<dt><tt>inputs[1]</tt>: tensor(int32, Linear)</dt>
<dd>Tensor of rank q>=1.</dd>

<dt><tt>inputs[2]</tt>: T</dt>
<dd>Tensor of rank q + r - indices_shape[-1] - 1.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Tensor of rank r >= 1.</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear), tensor(int32, Linear)

### NonMaxSuppression

#### Description

Filter out boxes has high IoU overlap with previously selected boxes or low score. Output the indices of valid boxes. Indices of invalid boxes will be filled with -1.

#### Parameters

| Type    | Parameter                    | Description                                                                                                                          |
| ------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `int`   | `center_point_box`           | 0 - the box data is supplied as \[y1, x1, y2, x2\], 1-the box data is supplied as \[x_center, y_center, width, height\].             |
| `int`   | `max_output_boxes_per_class` | The maximum number of boxes to be selected per batch per class. Default to 0, number of output boxes equal to number of input boxes. |
| `float` | `iou_threshold`              | The threshold for deciding whether boxes overlap too much with respect to IoU. Value range \[0, 1\]. Default to 0.                   |
| `float` | `score_threshold`            | The threshold for deciding when to remove boxes based on score.                                                                      |
| `int`   | `offset`                     | 0 or 1, boxes' width or height is (x2 - x1 + offset).                                                                                |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input boxes. 3-D tensor of shape (num_batches, spatial_dimension, 4).</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input scores. 3-D tensor of shape (num_batches, num_classes, spatial_dimension).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: tensor(int32, Linear)</dt>
<dd>Selected indices. 2-D tensor of shape (num_selected_indices, 3) as [[batch_index, class_index, box_index], ...].</dd>
<dd>num_selected_indices=num_batches* num_classes* min(max_output_boxes_per_class, spatial_dimension).</dd>
<dd>All invalid indices will be filled with -1.</dd>
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
<dt><tt>inputs[0]</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the numbers of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>inputs[1]</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, outH, outW, 2), where outH and outW is the height and width of offset and output. </dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, C, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### cummax

#### Description

Returns a namedtuple (`values`, `indices`) where `values` is the cumulative maximum of elements of `input` in the dimension `dim`. And `indices` is the index location of each maximum value found in the dimension `dim`.

#### Parameters

| Type  | Parameter | Description                             |
| ----- | --------- | --------------------------------------- |
| `int` | `dim`     | The dimension to do the operation over. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>The input tensor.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output values.</dd>
<dt><tt>outputs[1]</tt>: (int32, Linear)</dt>
<dd>Output indices.</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### cummin

#### Description

Returns a namedtuple (`values`, `indices`) where `values` is the cumulative minimum of elements of `input` in the dimension `dim`. And `indices` is the index location of each minimum value found in the dimension `dim`.

#### Parameters

| Type  | Parameter | Description                             |
| ----- | --------- | --------------------------------------- |
| `int` | `dim`     | The dimension to do the operation over. |

#### Inputs

<dl>
<dt><tt>inputs[0]</tt>: T</dt>
<dd>The input tensor.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output values.</dd>
<dt><tt>outputs[1]</tt>: (int32, Linear)</dt>
<dd>Output indices.</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVInstanceNormalization

#### Description

Carries out instance normalization as described in the paper https://arxiv.org/abs/1607.08022.

y = scale * (x - mean) / sqrt(variance + epsilon) + B, where mean and variance are computed per instance per channel.

#### Parameters

| Type    | Parameter | Description                                                          |
| ------- | --------- | -------------------------------------------------------------------- |
| `float` | `epsilon` | The epsilon value to use to avoid division by zero. Default is 1e-05 |

#### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input data tensor from the previous operator; dimensions for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are the height and the width of the data. For non image case, the dimensions are in the form of (N x C x D1 x D2 ... Dn), where N is the batch size.</dd>
<dt><tt>scale</tt>: T</dt>
<dd>The input 1-dimensional scale tensor of size C.</dd>
<dt><tt>B</tt>: T</dt>
<dd>The input 1-dimensional bias tensor of size C.</dd>
</dl>

#### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>The output tensor of the same shape as input.</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)

### MMCVModulatedDeformConv2d

#### Description

Perform Modulated Deformable Convolution on input feature, read [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168?from=timeline) for detail.

#### Parameters

| Type           | Parameter          | Description                                                                           |
| -------------- | ------------------ | ------------------------------------------------------------------------------------- |
| `list of ints` | `stride`           | The stride of the convolving kernel. (sH, sW)                                         |
| `list of ints` | `padding`          | Paddings on both sides of the input. (padH, padW)                                     |
| `list of ints` | `dilation`         | The spacing between kernel elements. (dH, dW)                                         |
| `int`          | `deformable_group` | Groups of deformable offset.                                                          |
| `int`          | `group`            | Split input into groups. `input_channel` should be divisible by the number of groups. |

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
<dd>Input weight; 1-D tensor of shape (output_channel).</dd>
</dl>

#### Outputs

<dl>
<dt><tt>outputs[0]</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, output_channel, outH, outW).</dd>
</dl>

#### Type Constraints

- T:tensor(float32, Linear)
