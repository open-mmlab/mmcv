"""
CommandLine:
    pytest tests/test_merge_cells.py
"""
import torch
import torch.nn.functional as F

from mmcv.ops.merge_cells import (BaseMergeCell, ConcatCell, GlobalPoolingCell,
                                  SumCell)


def test_sum_cell():
    inputs_x = torch.randn([2, 256, 14, 7])
    inputs_y = torch.randn([2, 256, 16, 16])
    sum_cell = SumCell(256, 256)
    output = sum_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert output.size() == inputs_x.size()
    output = sum_cell(inputs_x, inputs_y, out_size=inputs_y.shape[-2:])
    assert output.size() == inputs_y.size()
    output = sum_cell(inputs_x, inputs_y)
    assert output.size() == inputs_y.size()


def test_concat_cell():
    inputs_x = torch.randn([2, 256, 14, 7])
    inputs_y = torch.randn([2, 256, 16, 16])
    concat_cell = ConcatCell(256, 256)
    output = concat_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert output.size() == inputs_x.size()
    output = concat_cell(inputs_x, inputs_y, out_size=inputs_y.shape[-2:])
    assert output.size() == inputs_y.size()
    output = concat_cell(inputs_x, inputs_y)
    assert output.size() == inputs_y.size()


def test_global_pool_cell():
    inputs_x = torch.randn([2, 256, 14, 7])
    inputs_y = torch.randn([2, 256, 32, 32])
    gp_cell = GlobalPoolingCell(with_out_conv=False)
    gp_cell_out = gp_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert (gp_cell_out.size() == inputs_x.size())
    gp_cell = GlobalPoolingCell(256, 256)
    gp_cell_out = gp_cell(inputs_x, inputs_y, out_size=inputs_x.shape[-2:])
    assert (gp_cell_out.size() == inputs_x.size())


def test_resize_methods():
    inputs_x = torch.randn([2, 256, 128, 128])
    target_resize_sizes = [(128, 128), (256, 256)]
    resize_methods_list = ['nearest', 'bilinear']

    for method in resize_methods_list:
        merge_cell = BaseMergeCell(upsample_mode=method)
        for target_size in target_resize_sizes:
            merge_cell_out = merge_cell._resize(inputs_x, target_size)
            gt_out = F.interpolate(inputs_x, size=target_size, mode=method)
            assert merge_cell_out.equal(gt_out)
    """resize to a smaller size by which the inputs_x can't be divisible.
    And There're different downsample scale on dim H & dim W.
    """
    target_size = (14, 7)
    merge_cell = BaseMergeCell()
    merge_cell_out = merge_cell._resize(inputs_x, target_size)

    if inputs_x.shape[-2] % target_size[-2] != 0 or inputs_x.shape[
            -1] % target_size[-1] != 0:
        h, w = inputs_x.shape[-2:]
        h_t, w_t = target_size
        padding_h = (h // h_t + 1) * h_t - h
        padding_w = (w // w_t + 1) * w_t - w
        padding_left = padding_w // 2
        padding_right = padding_w - padding_left
        padding_top = padding_h // 2
        padding_bottom = padding_h - padding_top
        pad = (padding_left, padding_right, padding_top, padding_bottom)
        inputs_x = F.pad(inputs_x, pad, mode='constant', value=0.0)
    kernel_size = (inputs_x.shape[-2] // target_size[-2],
                   inputs_x.shape[-1] // target_size[-1])
    gt_out = F.max_pool2d(
        inputs_x, kernel_size=kernel_size, stride=kernel_size)
    assert (merge_cell_out == gt_out).all()
    assert merge_cell_out.shape[2:] == target_size
