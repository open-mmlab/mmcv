import numpy as np
import pytest
import torch


class Testnms(object):

    def test_nms_allclose(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import nms
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
        if not torch.cuda.is_available():
            return
        from mmcv.op import soft_nms
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

    def test_nms_match(self):
        if not torch.cuda.is_available():
            return
        from mmcv.op import nms, nms_match
        iou_thr = 0.6
        # empty input
        empty_dets = np.array([])
        assert len(nms_match(empty_dets, iou_thr)) == 0

        # non empty ndarray input
        np_dets = np.array(
            [[49.1, 32.4, 51.0, 35.9, 0.9], [49.3, 32.9, 51.0, 35.3, 0.9],
             [35.3, 11.5, 39.9, 14.5, 0.4], [35.2, 11.7, 39.7, 15.7, 0.3]],
            dtype=np.float32)
        np_groups = nms_match(np_dets, iou_thr)
        assert isinstance(np_groups[0], np.ndarray)
        assert len(np_groups) == 2
        tensor_dets = torch.from_numpy(np_dets)
        boxes = tensor_dets[:, :4]
        scores = tensor_dets[:, 4]
        nms_keep_inds = nms(boxes.contiguous(), scores.contiguous(),
                            iou_thr)[1]
        assert set([g[0].item()
                    for g in np_groups]) == set(nms_keep_inds.tolist())

        # non empty tensor input
        tensor_dets = torch.from_numpy(np_dets)
        tensor_groups = nms_match(tensor_dets, iou_thr)
        assert isinstance(tensor_groups[0], torch.Tensor)
        for i in range(len(tensor_groups)):
            assert np.equal(tensor_groups[i].numpy(), np_groups[i]).all()

        # input of wrong shape
        wrong_dets = np.zeros((2, 3))
        with pytest.raises(AssertionError):
            nms_match(wrong_dets, iou_thr)
