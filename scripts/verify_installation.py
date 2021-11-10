import numpy as np
import torch

from mmcv.ops import box_iou_rotated


def verify_installation():
    """Verify mmcv-full whether it has been installed successfully."""
    np_boxes1 = np.asarray(
        [[1.0, 1.0, 3.0, 4.0, 0.5], [2.0, 2.0, 3.0, 4.0, 0.6],
         [7.0, 7.0, 8.0, 8.0, 0.4]],
        dtype=np.float32)
    np_boxes2 = np.asarray(
        [[0.0, 2.0, 2.0, 5.0, 0.3], [2.0, 1.0, 3.0, 3.0, 0.5],
         [5.0, 5.0, 6.0, 7.0, 0.4]],
        dtype=np.float32)
    np_expect_ious = np.asarray(
        [[0.3708, 0.4351, 0.0000], [0.1104, 0.4487, 0.0424],
         [0.0000, 0.0000, 0.3622]],
        dtype=np.float32)
    np_expect_ious_aligned = np.asarray([0.3708, 0.4487, 0.3622],
                                        dtype=np.float32)
    boxes1 = torch.from_numpy(np_boxes1)
    boxes2 = torch.from_numpy(np_boxes2)

    # test mmcv-full with CPU ops
    ious = box_iou_rotated(boxes1, boxes2)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

    ious = box_iou_rotated(boxes1, boxes2, aligned=True)
    assert np.allclose(ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    # test mmcv-full with both CPU and CUDA ops
    if torch.cuda.is_available():
        boxes1 = boxes1.cuda()
        boxes2 = boxes2.cuda()

        ious = box_iou_rotated(boxes1, boxes2)
        assert np.allclose(ious.cpu().numpy(), np_expect_ious, atol=1e-4)

        ious = box_iou_rotated(boxes1, boxes2, aligned=True)
        assert np.allclose(
            ious.cpu().numpy(), np_expect_ious_aligned, atol=1e-4)

    print('mmcv-full has been installed successfully.')


if __name__ == '__main__':
    verify_installation()
