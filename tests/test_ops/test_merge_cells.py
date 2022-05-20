# Copyright (c) OpenMMLab. All rights reserved.
"""
CommandLine:
    pytest tests/test_merge_cells.py
"""
import math

import pytest
import torch
import torch.nn.functional as F

from mmcv.ops.merge_cells import (BaseMergeCell, ConcatCell, GlobalPoolingCell,
                                  SumCell)


# All size (14, 7) below is to test the situation that
# the input size can't be divisible by the target size.
@pytest.mark.parametrize(
    'inputs_x, inputs_y',
    [(torch.randn([2, 256, 16, 16]), torch.randn([2, 256, 32, 32])),
     (torch.randn([2, 256, 14, 7]), torch.randn([2, 256, 32, 32]))])
def test_sum_cell(inputs_x, inputs_y):
    sum_cell = SumCell(256, 256)
    output = sum_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert output.size() == inputs_x.size()
    output = sum_cell(inputs_x, inputs_y, out_size=inputs_y.shape[-2:])
    assert output.size() == inputs_y.size()
    output = sum_cell(inputs_x, inputs_y)
    assert output.size() == inputs_y.size()


@pytest.mark.parametrize(
    'inputs_x, inputs_y',
    [(torch.randn([2, 256, 16, 16]), torch.randn([2, 256, 32, 32])),
     (torch.randn([2, 256, 14, 7]), torch.randn([2, 256, 32, 32]))])
def test_concat_cell(inputs_x, inputs_y):
    concat_cell = ConcatCell(256, 256)
    output = concat_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert output.size() == inputs_x.size()
    output = concat_cell(inputs_x, inputs_y, out_size=inputs_y.shape[-2:])
    assert output.size() == inputs_y.size()
    output = concat_cell(inputs_x, inputs_y)
    assert output.size() == inputs_y.size()


@pytest.mark.parametrize(
    'inputs_x, inputs_y',
    [(torch.randn([2, 256, 16, 16]), torch.randn([2, 256, 32, 32])),
     (torch.randn([2, 256, 14, 7]), torch.randn([2, 256, 32, 32]))])
def test_global_pool_cell(inputs_x, inputs_y):
    gp_cell = GlobalPoolingCell(with_out_conv=False)
    gp_cell_out = gp_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert (gp_cell_out.size() == inputs_x.size())
    gp_cell = GlobalPoolingCell(256, 256)
    gp_cell_out = gp_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert (gp_cell_out.size() == inputs_x.size())


@pytest.mark.parametrize('target_size', [(256, 256), (128, 128), (64, 64),
                                         (14, 7)])
def test_resize_methods(target_size):
    inputs_x = torch.randn([2, 256, 128, 128])
    h, w = inputs_x.shape[-2:]
    target_h, target_w = target_size
    if (h <= target_h) or w <= target_w:
        rs_mode = 'upsample'
    else:
        rs_mode = 'downsample'

    if rs_mode == 'upsample':
        upsample_methods_list = ['nearest', 'bilinear']
        for method in upsample_methods_list:
            merge_cell = BaseMergeCell(upsample_mode=method)
            merge_cell_out = merge_cell._resize(inputs_x, target_size)
            gt_out = F.interpolate(inputs_x, size=target_size, mode=method)
            assert merge_cell_out.equal(gt_out)
    elif rs_mode == 'downsample':
        merge_cell = BaseMergeCell()
        merge_cell_out = merge_cell._resize(inputs_x, target_size)
        if h % target_h != 0 or w % target_w != 0:
            pad_h = math.ceil(h / target_h) * target_h - h
            pad_w = math.ceil(w / target_w) * target_w - w
            pad_l = pad_w // 2
            pad_r = pad_w - pad_l
            pad_t = pad_h // 2
            pad_b = pad_h - pad_t
            pad = (pad_l, pad_r, pad_t, pad_b)
            inputs_x = F.pad(inputs_x, pad, mode='constant', value=0.0)
        kernel_size = (inputs_x.shape[-2] // target_h,
                       inputs_x.shape[-1] // target_w)
        gt_out = F.max_pool2d(
            inputs_x, kernel_size=kernel_size, stride=kernel_size)
        print(merge_cell_out.shape, gt_out.shape)
        assert (merge_cell_out == gt_out).all()
        assert merge_cell_out.shape[-2:] == target_size
