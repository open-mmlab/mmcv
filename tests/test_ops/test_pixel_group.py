# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch


def test_pixel_group():
    from mmcv.ops import pixel_group
    np_score = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0],
                         [0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0],
                         [0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0],
                         [0, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]).astype(np.float32)
    np_mask = (np_score > 0.5)
    np_embedding = np.zeros((10, 10, 8)).astype(np.float32)
    np_embedding[:, :7] = 0.9
    np_embedding[:, 7:] = 10.0
    np_kernel_label = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0, 0, 2, 0],
                                [0, 0, 1, 1, 1, 0, 0, 0, 2, 0],
                                [0, 0, 1, 1, 1, 0, 0, 0, 2, 0],
                                [0, 0, 1, 1, 1, 0, 0, 0, 2, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0]]).astype(np.int32)
    np_kernel_contour = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 1, 1, 1, 0, 0, 0, 1, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                  [0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0]]).astype(np.uint8)
    kernel_region_num = 3
    distance_threshold = float(0.8)
    result = pixel_group(np_score, np_mask, np_embedding, np_kernel_label,
                         np_kernel_contour, kernel_region_num,
                         distance_threshold)
    gt_1 = [
        0.8999997973442078, 24.0, 1.0, 3.0, 2.0, 3.0, 3.0, 3.0, 4.0, 3.0, 5.0,
        3.0, 6.0, 3.0, 1.0, 4.0, 2.0, 4.0, 3.0, 4.0, 4.0, 4.0, 5.0, 4.0, 6.0,
        4.0, 1.0, 5.0, 2.0, 5.0, 3.0, 5.0, 4.0, 5.0, 5.0, 5.0, 6.0, 5.0, 1.0,
        6.0, 2.0, 6.0, 3.0, 6.0, 4.0, 6.0, 5.0, 6.0, 6.0, 6.0
    ]

    gt_2 = [
        0.9000000357627869, 8.0, 7.0, 3.0, 8.0, 3.0, 7.0, 4.0, 8.0, 4.0, 7.0,
        5.0, 8.0, 5.0, 7.0, 6.0, 8.0, 6.0
    ]

    assert np.allclose(result[0], [0, 0])
    assert np.allclose(result[1], gt_1)
    assert np.allclose(result[2], gt_2)

    # test torch Tensor
    np_score_t = torch.from_numpy(np_score)
    np_mask_t = torch.from_numpy(np_mask)
    np_embedding_t = torch.from_numpy(np_embedding)
    np_kernel_label_t = torch.from_numpy(np_kernel_label)
    np_kernel_contour_t = torch.from_numpy(np_kernel_contour)

    result = pixel_group(np_score_t, np_mask_t, np_embedding_t,
                         np_kernel_label_t, np_kernel_contour_t,
                         kernel_region_num, distance_threshold)

    assert np.allclose(result[0], [0, 0])
    assert np.allclose(result[1], gt_1)
    assert np.allclose(result[2], gt_2)
