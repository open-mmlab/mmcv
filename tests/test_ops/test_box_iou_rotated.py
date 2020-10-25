import numpy as np
import torch


class TestBoxIoURotated(object):

    def test_box_iou_rotated(self):
        if not torch.cuda.is_available():
            return
        from mmcv.ops import box_iou_rotated
        b1 = torch.tensor([[1.0, 1.0, 3.0, 4.0], [2.0, 2.0, 3.0, 4.0],
                           [7.0, 7.0, 8.0, 8.0]], dtype=torch.float32).cuda()
        b2 = torch.tensor(
            [[0.0, 2.0, 2.0, 5.0], [2.0, 1.0, 3.0, 3.0]], dtype=torch.float32).cuda()
        expect_output = torch.tensor([[0.2715, 0.0000],
                                      [0.1396, 0.0000],
                                      [0.0000, 0.0000]], dtype=torch.float32).cuda()
        output = box_iou_rotated(b1, b2)
        assert np.allclose(output.cpu().numpy(), expect_output.cpu().numpy(), atol=1e-4)
