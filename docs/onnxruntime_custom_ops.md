# Onnxruntime Custom Ops

<!-- TOC -->

- [Onnxruntime Custom Ops](#onnxruntime-custom-ops)
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

<!-- TOC -->

## SoftNMS

### Description

Perform soft NMS on `boxes` with `scores`. Read [Soft-NMS -- Improving Object Detection With One Line of Code](https://arxiv.org/abs/1704.04503) for detail.

### Parameters

| Type    | Parameter       | Description                                                    |
| ------- | --------------- | -------------------------------------------------------------- |
| `float` | `iou_threshold` | IoU threshold for NMS                                          |
| `float` | `sigma`         | hyperparameter for gaussian method                             |
| `float` | `min_score`     | score filter threshold                                         |
| `int`   | `method`        | method to do the nms, (0: `naive`, 1: `linear`, 2: `gaussian`) |
| `int`   | `offset`        | `boxes` width or height is (x2 - x1 + offset). (0 or 1)        |

### Inputs

<dl>
<dt><tt>boxes</tt>: T</dt>
<dd>Input boxes. 2-D tensor of shape (N, 4). N is the number of boxes.</dd>
<dt><tt>scores</tt>: T</dt>
<dd>Input scores. 1-D tensor of shape (N, ).</dd>
</dl>

### Outputs

<dl>
<dt><tt>dets</tt>: tensor(int64)</dt>
<dd>Output boxes and scores. 2-D tensor of shape (num_valid_boxes, 5), [[x1, y1, x2, y2, score], ...]. num_valid_boxes is the number of valid boxes.</dd>
<dt><tt>indices</tt>: T</dt>
<dd>Output indices. 1-D tensor of shape (num_valid_boxes, ).</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## RoIAlign

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

## NMS

### Description

Filter out boxes has high IoU overlap with previously selected boxes.

### Parameters

| Type    | Parameter       | Description                                                                                                      |
| ------- | --------------- | ---------------------------------------------------------------------------------------------------------------- |
| `float` | `iou_threshold` | The threshold for deciding whether boxes overlap too much with respect to IoU. Value range [0, 1]. Default to 0. |
| `int`   | `offset`        | 0 or 1, boxes' width or height is (x2 - x1 + offset).                                                            |

### Inputs

<dl>
<dt><tt>bboxes</tt>: T</dt>
<dd>Input boxes. 2-D tensor of shape (num_boxes, 4). num_boxes is the number of input boxes.</dd>
<dt><tt>scores</tt>: T</dt>
<dd>Input scores. 1-D tensor of shape (num_boxes, ).</dd>
</dl>

### Outputs

<dl>
<dt><tt>indices</tt>: tensor(int32, Linear)</dt>
<dd>Selected indices. 1-D tensor of shape (num_valid_boxes, ). num_valid_boxes is the number of valid boxes.</dd>
</dl>

### Type Constraints

- T:tensor(float32)

## grid_sampler

### Description

Perform sample from `input` with pixel locations from `grid`.

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
<dd>Input offset; 4-D tensor of shape (N, outH, outW, 2), where outH and outW is the height and width of offset and output. </dd>
</dl>

### Outputs

<dl>
<dt><tt>output</tt>: T</dt>
<dd>Output feature; 4-D tensor of shape (N, C, outH, outW).</dd>
</dl>

### Type Constraints

- T:tensor(float32, Linear)
