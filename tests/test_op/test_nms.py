import numpy as np
import torch

from mmcv.op import nms, soft_nms


class Testnms(object):

    def test_nms_allclose(self):
        np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                             [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                            dtype=np.float32)
        np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)
        np_inds = np.array([1, 0, 3])
        np_dets = np.array([[3.0, 6.0, 9.0, 11.0, 0.9],
                            [6.0, 3.0, 8.0, 7.0, 0.6],
                            [1.0, 4.0, 13.0, 7.0, 0.2]])
        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)
        dets, inds = nms(boxes, scores, iou_threshold=0.3, offset=0)
        assert np.allclose(dets, np_dets)  # test cpu
        assert np.allclose(inds, np_inds)  # test cpu
        dets, inds = nms(
            boxes.cuda(), scores.cuda(), iou_threshold=0.3, offset=0)
        assert np.allclose(dets.cpu().numpy(), np_dets)  # test gpu
        assert np.allclose(inds.cpu().numpy(), np_inds)  # test gpu

    def test_softnms_allclose(self):
        np_boxes = np.array([[6.0, 3.0, 8.0, 7.0], [3.0, 6.0, 9.0, 11.0],
                             [3.0, 7.0, 10.0, 12.0], [1.0, 4.0, 13.0, 7.0]],
                            dtype=np.float32)
        np_scores = np.array([0.6, 0.9, 0.7, 0.2], dtype=np.float32)

        np_output = {
            'linear': {
                'dets':
                np.array(
                    [[3., 6., 9., 11., 0.9], [6., 3., 8., 7., 0.6],
                     [3., 7., 10., 12., 0.29024392], [1., 4., 13., 7., 0.2]],
                    dtype=np.float32),
                'inds':
                np.array([1, 0, 2, 3], dtype=np.int64)
            },
            'gaussian': {
                'dets':
                np.array([[3., 6., 9., 11., 0.9], [6., 3., 8., 7., 0.59630775],
                          [3., 7., 10., 12., 0.35275510],
                          [1., 4., 13., 7., 0.18650459]],
                         dtype=np.float32),
                'inds':
                np.array([1, 0, 2, 3], dtype=np.int64)
            },
            'naive': {
                'dets':
                np.array([[3., 6., 9., 11., 0.9], [6., 3., 8., 7., 0.6],
                          [1., 4., 13., 7., 0.2]],
                         dtype=np.float32),
                'inds':
                np.array([1, 0, 3], dtype=np.int64)
            }
        }

        boxes = torch.from_numpy(np_boxes)
        scores = torch.from_numpy(np_scores)

        configs = [[0.3, 0.5, 0.01, 'linear'], [0.3, 0.5, 0.01, 'gaussian'],
                   [0.3, 0.5, 0.01, 'naive']]

        for iou, sig, mscore, m in configs:
            dets, inds = soft_nms(
                boxes,
                scores,
                iou_threshold=iou,
                sigma=sig,
                min_score=mscore,
                method=m)
            assert np.allclose(dets.cpu().numpy(), np_output[m]['dets'])
            assert np.allclose(inds.cpu().numpy(), np_output[m]['inds'])

        if torch.__version__ != 'parrots':
            boxes = boxes.cuda()
            scores = scores.cuda()
            for iou, sig, mscore, m in configs:
                dets, inds = soft_nms(
                    boxes,
                    scores,
                    iou_threshold=iou,
                    sigma=sig,
                    min_score=mscore,
                    method=m)
                assert np.allclose(dets.cpu().numpy(), np_output[m]['dets'])
                assert np.allclose(inds.cpu().numpy(), np_output[m]['inds'])
