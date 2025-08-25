# Copyright (c) VBTI. All rights reserved.

import numpy as np
import pytest
import torch

from mmcv.ops import line_nms
from mmcv.utils import IS_CUDA_AVAILABLE


# Test case 1: Simple non-overlapping lines
def create_line(x1, y1, start_pos, length, offsets):
    """Create a line with 77 features:
      [x1, y1, start_pos, ?, length, ...72_offsets]"""
    line = np.zeros(77)
    line[0] = x1  # x coordinate
    line[1] = y1  # y coordinate
    line[2] = start_pos  # starting position
    line[3] = 0  # placeholder (from your CUDA code)
    line[4] = length  # length
    line[5:] = np.array(offsets[:72])  # 72 offset values
    return line


@pytest.fixture
def lines():
    # Create test lines with different characteristics
    lines = []
    scores = []

    # Line 1: Horizontal line at top
    offsets1 = [10.0] * 72  # Constant offset
    line1 = create_line(100, 50, 0.1, 30, offsets1)
    lines.append(line1)
    scores.append(0.9)

    # Line 2: Similar horizontal line (should be suppressed)
    offsets2 = [12.0] * 72  # Slightly different but overlapping
    line2 = create_line(110, 52, 0.12, 28, offsets2)
    lines.append(line2)
    scores.append(0.7)

    # Line 3: Vertical line (non-overlapping)
    offsets3 = list(range(72))  # Increasing offsets
    line3 = create_line(200, 100, 0.2, 40, offsets3)
    lines.append(line3)
    scores.append(0.8)

    # Line 4: Diagonal line (non-overlapping)
    offsets4 = [i * 0.5 for i in range(72)]
    line4 = create_line(300, 150, 0.3, 35, offsets4)
    lines.append(line4)
    scores.append(0.6)

    # Line 5: Another line similar to line 3 (should be suppressed)
    offsets5 = [i + 1 for i in range(72)]
    line5 = create_line(205, 102, 0.21, 38, offsets5)
    lines.append(line5)
    scores.append(0.5)

    boxes = np.stack(lines)
    scores = np.array(scores)

    return boxes, scores


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason='requires CUDA support')
@pytest.mark.parametrize('overlap_thresh,top_k,n_expected', [
    (0.5, 5, 4),
    (0, 5, 5),
    (1, 5, 4),
    (1, 1, 1),
])
def test_line_nms(lines, overlap_thresh, top_k, n_expected):

    np_lines, np_scores = lines
    device = 'cuda'

    lines_ = torch.from_numpy(np_lines).to(device)
    scores = torch.from_numpy(np_scores).to(device)
    _, num_to_keep, _ = line_nms(lines_, scores, overlap_thresh, top_k)

    assert num_to_keep == n_expected


@pytest.mark.parametrize('device', [
    pytest.param(
        'cuda',
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE, reason='requires CUDA support'))
])
def test_line_nms_real_data(device):
    data = np.load('./tests/data/line_nms_input.npz')
    np_lines = data['lines']
    np_scores = data['scores']

    # boxes, idx
    overlap_thresh = 0.5
    top_k = 5

    lines = torch.from_numpy(np_lines).to(device)
    scores = torch.from_numpy(np_scores).to(device)

    keep, num_to_keep, parent_obj_idx = line_nms(lines, scores, overlap_thresh,
                                                 top_k)

    assert num_to_keep <= top_k

    expected_keep = [
        150, 135, 36, 168, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0
    ]
    expected_num_to_keep = 5
    expected_parent_object_index = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]

    assert np.array_equal(keep.cpu().numpy(), np.array(expected_keep))
    assert np.array_equal(parent_obj_idx.cpu().numpy(),
                          np.array(expected_parent_object_index))
    assert expected_num_to_keep == num_to_keep
