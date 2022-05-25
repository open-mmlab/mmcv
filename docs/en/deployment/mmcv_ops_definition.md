# MMCV Operators

To make custom operators in MMCV more standard, precise definitions of each operator are listed in this document.

<!-- TOC -->

- [MMCV Operators](#mmcv-operators)
  - [MMCVBorderAlign](#mmcvborderalign)
    - [Description](#description)
    - [Parameters](#parameters)
    - [Inputs](#inputs)
    - [Outputs](#outputs)
    - [Type Constraints](#type-constraints)
  - [MMCVCARAFE](#mmcvcarafe)
    - [Description](#description-1)
    - [Parameters](#parameters-1)
    - [Inputs](#inputs-1)
    - [Outputs](#outputs-1)
    - [Type Constraints](#type-constraints-1)
  - [MMCVCAWeight](#mmcvcaweight)
    - [Description](#description-2)
    - [Parameters](#parameters-2)
    - [Inputs](#inputs-2)
    - [Outputs](#outputs-2)
    - [Type Constraints](#type-constraints-2)
  - [MMCVCAMap](#mmcvcamap)
    - [Description](#description-3)
    - [Parameters](#parameters-3)
    - [Inputs](#inputs-3)
    - [Outputs](#outputs-3)
    - [Type Constraints](#type-constraints-3)
  - [MMCVCornerPool](#mmcvcornerpool)
    - [Description](#description-4)
    - [Parameters](#parameters-4)
    - [Inputs](#inputs-4)
    - [Outputs](#outputs-4)
    - [Type Constraints](#type-constraints-4)
  - [MMCVDeformConv2d](#mmcvdeformconv2d)
    - [Description](#description-5)
    - [Parameters](#parameters-5)
    - [Inputs](#inputs-5)
    - [Outputs](#outputs-5)
    - [Type Constraints](#type-constraints-5)
  - [MMCVModulatedDeformConv2d](#mmcvmodulateddeformconv2d)
    - [Description](#description-6)
    - [Parameters](#parameters-6)
    - [Inputs](#inputs-6)
    - [Outputs](#outputs-6)
    - [Type Constraints](#type-constraints-6)
  - [MMCVDeformRoIPool](#mmcvdeformroipool)
    - [Description](#description-7)
    - [Parameters](#parameters-7)
    - [Inputs](#inputs-7)
    - [Outputs](#outputs-7)
    - [Type Constraints](#type-constraints-7)
  - [MMCVMaskedConv2d](#mmcvmaskedconv2d)
    - [Description](#description-8)
    - [Parameters](#parameters-8)
    - [Inputs](#inputs-8)
    - [Outputs](#outputs-8)
    - [Type Constraints](#type-constraints-8)
  - [MMCVPSAMask](#mmcvpsamask)
    - [Description](#description-9)
    - [Parameters](#parameters-9)
    - [Inputs](#inputs-9)
    - [Outputs](#outputs-9)
    - [Type Constraints](#type-constraints-9)
  - [NonMaxSuppression](#nonmaxsuppression)
    - [Description](#description-10)
    - [Parameters](#parameters-10)
    - [Inputs](#inputs-10)
    - [Outputs](#outputs-10)
    - [Type Constraints](#type-constraints-10)
  - [MMCVRoIAlign](#mmcvroialign)
    - [Description](#description-11)
    - [Parameters](#parameters-11)
    - [Inputs](#inputs-11)
    - [Outputs](#outputs-11)
    - [Type Constraints](#type-constraints-11)
  - [MMCVRoIAlignRotated](#mmcvroialignrotated)
    - [Description](#description-12)
    - [Parameters](#parameters-12)
    - [Inputs](#inputs-12)
    - [Outputs](#outputs-12)
    - [Type Constraints](#type-constraints-12)
  - [grid_sampler\*](#grid_sampler)
    - [Description](#description-13)
    - [Parameters](#parameters-13)
    - [Inputs](#inputs-13)
    - [Outputs](#outputs-13)
    - [Type Constraints](#type-constraints-13)
  - [cummax\*](#cummax)
    - [Description](#description-14)
    - [Parameters](#parameters-14)
    - [Inputs](#inputs-14)
    - [Outputs](#outputs-14)
    - [Type Constraints](#type-constraints-14)
  - [cummin\*](#cummin)
    - [Description](#description-15)
    - [Parameters](#parameters-15)
    - [Inputs](#inputs-15)
    - [Outputs](#outputs-15)
    - [Type Constraints](#type-constraints-15)
  - [Reminders](#reminders)

<!-- TOC -->

## MMCVBorderAlign

### Description

Applies `border_align` over the input feature based on predicted bboxes.

For each border line (e.g. top, left, bottom or right) of each box,
border_align does the following:

- uniformly samples `pool_size`+1 positions on this line, involving the start and end points.
- the corresponding features on these points are computed by bilinear interpolation.
- max pooling over all the `pool_size`+1 positions are used for computing pooled feature.

Read [BorderDet: Border Feature for Dense Object Detection](ttps://arxiv.org/abs/2007.11056) for more detailed information.

### Parameters

| Type  | Parameter   | Description                                                                         |
| ----- | ----------- | ----------------------------------------------------------------------------------- |
| `int` | `pool_size` | number of positions sampled over the boxes' borders(e.g. top, bottom, left, right). |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Features with shape [N,4C,H,W]. Channels ranged in [0,C), [C,2C), [2C,3C), [3C,4C) represent the top, left, bottom, right features respectively</dd>
<dt><tt>boxes</tt>: T</dt>
<dd>Boxes with shape [N,H*W,4]. Coordinate format (x1,y1,x2,y2).</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Pooled features with shape [N,C,H*W,4]. The order is(top,left,bottom,right) for the last dimension.</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVCARAFE

### Description

CARAFE operator performs feature upsampling.

Read [CARAFE: Content-Aware ReAssembly of FEatures](https://arxiv.org/abs/1905.02188) for more detailed information.

### Parameters

| Type    | Parameter      | Description                                   |
| ------- | -------------- | --------------------------------------------- |
| `int`   | `kernel_size`  | reassemble kernel size, should be odd integer |
| `int`   | `group_size`   | reassemble group size                         |
| `float` | `scale_factor` | upsample ratio(>=1)                           |

### Inputs

<dl>
<dt><tt>features</tt>: T</dt>
<dd>Input features. 4-D tensor of shape (N, C, H, W). N is the batch size.</dd>
<dt><tt>masks</tt>: T</dt>
<dd>The input mask</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>The upsampled features. 4-D tensor of shape (N, C, H * scale_factor, W * scale_factor). N is the batch size.</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVCAWeight

### Description

Operator for Criss-Cross Attention
Read [CCNet: Criss-Cross Attention for SemanticSegmentation](https://arxiv.org/pdf/1811.11721.pdf) for more detailed information.

### Parameters

None

### Inputs

<dl>
<dt><tt>t</tt>: T</dt>
<dd>The query matrix of shape (N, C', H, W).</dd>
<dt><tt>f</tt>: T</dt>
<dd>The key matrix of shape (N, C', H, W).</dd>
</dl>

### Outputs

<dl>
<dt><tt>weight</tt>: T</dt>
<dd>The attention map of shape (N, H+W-1, H, W).</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVCAMap

### Description

Operator for Criss-Cross Attention
Read [CCNet: Criss-Cross Attention for SemanticSegmentation](https://arxiv.org/pdf/1811.11721.pdf) for more detailed information.

### Parameters

None

### Inputs

<dl>
<dt><tt>weight</tt>: T</dt>
<dd>Output from the operator MMCVCAWeight.</dd>
<dt><tt>value</tt>: T</dt>
<dd>The value matrix of shape (N, C, H, W).</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output tensor of aggregated contextual information</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVCornerPool

### Description

Perform CornerPool on `input` features. Read [CornerNet -- Detecting Objects as Paired Keypoints](https://arxiv.org/abs/1808.01244) for more details.

### Parameters

| Type  | Parameter | Description                                                      |
| ----- | --------- | ---------------------------------------------------------------- |
| `int` | `mode`    | corner pool mode, (0: `top`, 1: `bottom`, 2: `left`, 3: `right`) |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input features. 4-D tensor of shape (N, C, H, W). N is the batch size.</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>The pooled features. 4-D tensor of shape (N, C, H, W).</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVDeformConv2d

### Description

Applies a deformable 2D convolution over an input signal composed of several input planes.

Read [Deformable Convolutional Networks](https://arxiv.org/pdf/1703.06211.pdf) for detail.

### Parameters

| Type           | Parameter           | Description                                                                                                       |
| -------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `list of ints` | `stride`            | The stride of the convolving kernel, (sH, sW). Defaults to `(1, 1)`.                                              |
| `list of ints` | `padding`           | Paddings on both sides of the input, (padH, padW).  Defaults to `(0, 0)`.                                         |
| `list of ints` | `dilation`          | The spacing between kernel elements (dH, dW). Defaults to `(1, 1)`.                                               |
| `int`          | `groups`            | Split input into groups. `input_channel` should be divisible by the number of groups. Defaults to `1`.            |
| `int`          | `deformable_groups` | Groups of deformable offset. Defaults to `1`.                                                                     |
| `int`          | `bias`              | Whether to add a learnable bias to the output. `0` stands for `False` and `1` stands for `True`. Defaults to `0`. |
| `int`          | `im2col_step`       | Groups of deformable offset. Defaults to `32`.                                                                    |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the number of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>offset</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, deformable_group* 2* kH* kW, outH, outW), where kH and kW are the height and width of weight, outH and outW is the height and width of offset and output.</dd>
<dt><tt>weight</tt>: T</dt>
<dd>Input weight; 4-D tensor of shape (output_channel, input_channel, kH, kW).</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, output_channel, outH, outW).</dd>
</dl>

### Type Constraints

- T:tensor(float32, Linear)

## MMCVModulatedDeformConv2d

### Description

Perform Modulated Deformable Convolution on input feature, read [Deformable ConvNets v2: More Deformable, Better Results](https://arxiv.org/abs/1811.11168?from=timeline) for detail.

### Parameters

| Type           | Parameter           | Description                                                                           |
| -------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `list of ints` | `stride`            | The stride of the convolving kernel. (sH, sW)                                         |
| `list of ints` | `padding`           | Paddings on both sides of the input. (padH, padW)                                     |
| `list of ints` | `dilation`          | The spacing between kernel elements. (dH, dW)                                         |
| `int`          | `deformable_groups` | Groups of deformable offset.                                                          |
| `int`          | `groups`            | Split input into groups. `input_channel` should be divisible by the number of groups. |

### Inputs

<dl>
<dt><tt>feature</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the number of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>offset</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, deformable_group* 2* kH* kW, outH, outW), where kH and kW are the height and width of weight, outH and outW are the height and width of offset and output.</dd>
<dt><tt>mask</tt>: T</dt>
<dd>Input mask; 4-D tensor of shape (N, deformable_group* kH* kW, outH, outW), where kH and kW are the height and width of weight, outH and outW are the height and width of offset and output.</dd>
<dt><tt>weight]</tt>: T</dt>
<dd>Input weight; 4-D tensor of shape (output_channel, input_channel, kH, kW).</dd>
<dt><tt>bias</tt>: T, optional</dt>
<dd>Input bias; 1-D tensor of shape (output_channel).</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, output_channel, outH, outW).</dd>
</dl>

### Type Constraints

- T:tensor(float32, Linear)

## MMCVDeformRoIPool

### Description

Deformable roi pooling layer

### Parameters

| Type    | Parameter        | Description                                                                                                   |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `int`   | `output_height`  | height of output roi                                                                                          |
| `int`   | `output_width`   | width of output roi                                                                                           |
| `float` | `spatial_scale`  | used to scale the input boxes                                                                                 |
| `int`   | `sampling_ratio` | number of input samples to take for each output sample. `0` means to take samples densely for current models. |
| `float` | `gamma`          | gamma                                                                                                         |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
<dt><tt>rois</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 5) given as [[batch_index, x1, y1, x2, y2], ...]. The RoIs' coordinates are the coordinate system of input.</dd>
<dt><tt>offset</tt>: T</dt>
<dd>offset of height and width. Defaults to a tensor of zero</dd>
</dl>

### Outputs

<dl>
<dt><tt>feat</tt>: T</dt>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element feat[r-1] is a pooled feature map corresponding to the r-th RoI RoIs[r-1].<dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVMaskedConv2d

### Description

Performs a masked 2D convolution from PixelRNN
Read [Pixel Recurrent Neural Networks](https://arxiv.org/abs/1601.06759) for more detailed information.

### Parameters

| Type           | Parameter | Description                                                                      |
| -------------- | --------- | -------------------------------------------------------------------------------- |
| `list of ints` | `stride`  | The stride of the convolving kernel. (sH, sW). **Only support stride=1 in mmcv** |
| `list of ints` | `padding` | Paddings on both sides of the input. (padH, padW). Defaults to `(0, 0)`.         |

### Inputs

<dl>
<dt><tt>features</tt>: T</dt>
<dd>Input features; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
<dt><tt>mask</tt>: T</dt>
<dd>Input mask; 3D tensor of shape (N, H, W)</dd>
<dt><tt>weight</tt>: T</dt>
<dd>The learnable weights of the module</dd>
<dt><tt>bias</tt>: T</dt>
<dd>The learnable bias of the module</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>The output convolved feature</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVPSAMask

### Description

An operator from PSANet.

Read [PSANet: Point-wise Spatial Attention Network for Scene Parsing](https://hszhao.github.io/papers/eccv18_psanet.pdf) for more detailed information.

### Parameters

| Type           | Parameter   | Description                                  |
| -------------- | ----------- | -------------------------------------------- |
| `int`          | `psa_type`  | `0` means collect and `1` means `distribute` |
| `list of ints` | `mask_size` | The size of mask                             |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output tensor of shape (N, H * W, H, W)</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## NonMaxSuppression

### Description

Filter out boxes has high IoU overlap with previously selected boxes or low score. Output the indices of valid boxes.

Note this definition is slightly different with [onnx: NonMaxSuppression](https://github.com/onnx/onnx/blob/master/docs/Operators.md#nonmaxsuppression)

### Parameters

| Type    | Parameter                    | Description                                                                                                                          |
| ------- | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `int`   | `center_point_box`           | 0 - the box data is supplied as \[y1, x1, y2, x2\], 1-the box data is supplied as \[x_center, y_center, width, height\].             |
| `int`   | `max_output_boxes_per_class` | The maximum number of boxes to be selected per batch per class. Default to 0, number of output boxes equal to number of input boxes. |
| `float` | `iou_threshold`              | The threshold for deciding whether boxes overlap too much with respect to IoU. Value range \[0, 1\]. Default to 0.                   |
| `float` | `score_threshold`            | The threshold for deciding when to remove boxes based on score.                                                                      |
| `int`   | `offset`                     | 0 or 1, boxes' width or height is (x2 - x1 + offset).                                                                                |

### Inputs

<dl>
<dt><tt>boxes</tt>: T</dt>
<dd>Input boxes. 3-D tensor of shape (num_batches, spatial_dimension, 4).</dd>
<dt><tt>scores</tt>: T</dt>
<dd>Input scores. 3-D tensor of shape (num_batches, num_classes, spatial_dimension).</dd>
</dl>

### Outputs

<dl>
<dt><tt>indices</tt>: tensor(int32, Linear)</dt>
<dd>Selected indices. 2-D tensor of shape (num_selected_indices, 3) as [[batch_index, class_index, box_index], ...].</dd>
<dd>num_selected_indices=num_batches* num_classes* min(max_output_boxes_per_class, spatial_dimension).</dd>
<dd>All invalid indices will be filled with -1.</dd>
</dl>

### Type Constraints

- T:tensor(float32, Linear)

## MMCVRoIAlign

### Description

Perform RoIAlign on output feature, used in bbox_head of most two-stage detectors.

### Parameters

| Type    | Parameter        | Description                                                                                                   |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `int`   | `output_height`  | height of output roi                                                                                          |
| `int`   | `output_width`   | width of output roi                                                                                           |
| `float` | `spatial_scale`  | used to scale the input boxes                                                                                 |
| `int`   | `sampling_ratio` | number of input samples to take for each output sample. `0` means to take samples densely for current models. |
| `str`   | `mode`           | pooling mode in each bin. `avg` or `max`                                                                      |
| `int`   | `aligned`        | If `aligned=0`, use the legacy implementation in MMDetection. Else, align the results more perfectly.         |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W), where N is the batch size, C is the numbers of channels, H and W are the height and width of the data.</dd>
<dt><tt>rois</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 5) given as [[batch_index, x1, y1, x2, y2], ...]. The RoIs' coordinates are the coordinate system of input.</dd>
</dl>

### Outputs

<dl>
<dt><tt>feat</tt>: T</dt>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element feat[r-1] is a pooled feature map corresponding to the r-th RoI RoIs[r-1].<dd>
</dl>

### Type Constraints

- T:tensor(float32)

## MMCVRoIAlignRotated

### Description

Perform RoI align pooling for rotated proposals

### Parameters

| Type    | Parameter        | Description                                                                                                   |
| ------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `int`   | `output_height`  | height of output roi                                                                                          |
| `int`   | `output_width`   | width of output roi                                                                                           |
| `float` | `spatial_scale`  | used to scale the input boxes                                                                                 |
| `int`   | `sampling_ratio` | number of input samples to take for each output sample. `0` means to take samples densely for current models. |
| `str`   | `mode`           | pooling mode in each bin. `avg` or `max`                                                                      |
| `int`   | `aligned`        | If `aligned=0`, use the legacy implementation in MMDetection. Else, align the results more perfectly.         |
| `int`   | `clockwise`      | If `aligned=0`, use the legacy implementation in MMDetection. Else, align the results more perfectly.         |

### Inputs

<dl>
<dt><tt>features</tt>: T</dt>
<dd>Input feature map; 4D tensor of shape (N, C, H, W)</dd>
<dt><tt>rois</tt>: T</dt>
<dd>RoIs (Regions of Interest) to pool over; 2-D tensor of shape (num_rois, 5) given as [[batch_index, x1, y1, x2, y2], ...]. The RoIs' coordinates are the coordinate system of input.</dd>
</dl>

### Outputs

<dl>
<dd>RoI pooled output, 4-D tensor of shape (num_rois, C, output_height, output_width). The r-th batch element feat[r-1] is a pooled feature map corresponding to the r-th RoI RoIs[r-1].<dd>
</dl>

### Type Constraints

- T:tensor(float32)

## grid_sampler\*

### Description

Perform sample from `input` with pixel locations from `grid`.

Check [torch.nn.functional.grid_sample](https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html?highlight=grid_sample#torch.nn.functional.grid_sample) for more information.

### Parameters

| Type  | Parameter            | Description                                                                                                                                                                                                                                                                                     |
| ----- | -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `int` | `interpolation_mode` | Interpolation mode to calculate output values. (0: `bilinear` , 1: `nearest`)                                                                                                                                                                                                                   |
| `int` | `padding_mode`       | Padding mode for outside grid values. (0: `zeros`, 1: `border`, 2: `reflection`)                                                                                                                                                                                                                |
| `int` | `align_corners`      | If `align_corners=1`, the extrema (`-1` and `1`) are considered as referring to the center points of the input's corner pixels. If `align_corners=0`, they are instead considered as referring to the corner points of the input's corner pixels, making the sampling more resolution agnostic. |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>Input feature; 4-D tensor of shape (N, C, inH, inW), where N is the batch size, C is the numbers of channels, inH and inW are the height and width of the data.</dd>
<dt><tt>grid</tt>: T</dt>
<dd>Input offset; 4-D tensor of shape (N, outH, outW, 2), where outH and outW are the height and width of offset and output. </dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, C, outH, outW).</dd>
</dl>

### Type Constraints

- T:tensor(float32, Linear)

## cummax\*

### Description

Returns a tuple (`values`, `indices`) where `values` is the cumulative maximum elements of `input` in the dimension `dim`. And `indices` is the index location of each maximum value found in the dimension `dim`. Read [torch.cummax](https://pytorch.org/docs/stable/generated/torch.cummax.html) for more details.

### Parameters

| Type  | Parameter | Description                            |
| ----- | --------- | -------------------------------------- |
| `int` | `dim`     | the dimension to do the operation over |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>The input tensor with various shapes. Tensor with empty element is also supported.</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output the cumulative maximum elements of `input` in the dimension `dim`, with the same shape and dtype as `input`.</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>Output the index location of each cumulative maximum value found in the dimension `dim`, with the same shape as `input`.</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## cummin\*

### Description

Returns a tuple (`values`, `indices`) where `values` is the cumulative minimum elements of `input` in the dimension `dim`. And `indices` is the index location of each minimum value found in the dimension `dim`. Read [torch.cummin](https://pytorch.org/docs/stable/generated/torch.cummin.html) for more details.

### Parameters

| Type  | Parameter | Description                            |
| ----- | --------- | -------------------------------------- |
| `int` | `dim`     | the dimension to do the operation over |

### Inputs

<dl>
<dt><tt>input</tt>: T</dt>
<dd>The input tensor with various shapes. Tensor with empty element is also supported.</dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output the cumulative minimum elements of `input` in the dimension `dim`, with the same shape and dtype as `input`.</dd>
<dt><tt>indices</tt>: tensor(int64)</dt>
<dd>Output the index location of each cumulative minimum value found in the dimension `dim`, with the same shape as `input`.</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## Reminders

- Operators endwith `*` are defined in Torch and are included here for the conversion to ONNX.
