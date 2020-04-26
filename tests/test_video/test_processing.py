# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import tempfile

import mmcv


class TestVideoEditor:

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), '../data/test.mp4')
        cls.num_frames = 168

    def test_cut_concat_video(self):
        part1_file = osp.join(tempfile.gettempdir(), '.mmcv_test1.mp4')
        part2_file = osp.join(tempfile.gettempdir(), '.mmcv_test2.mp4')
        mmcv.cut_video(self.video_path, part1_file, end=3, vcodec='h264')
        mmcv.cut_video(self.video_path, part2_file, start=3, vcodec='h264')
        v1 = mmcv.VideoReader(part1_file)
        v2 = mmcv.VideoReader(part2_file)
        assert len(v1) == 75
        assert len(v2) == self.num_frames - 75

        out_file = osp.join(tempfile.gettempdir(), '.mmcv_test.mp4')
        mmcv.concat_video([part1_file, part2_file], out_file)
        v = mmcv.VideoReader(out_file)
        assert len(v) == self.num_frames
        os.remove(part1_file)
        os.remove(part2_file)
        os.remove(out_file)

    def test_resize_video(self):
        out_file = osp.join(tempfile.gettempdir(), '.mmcv_test.mp4')
        mmcv.resize_video(self.video_path, out_file, (200, 100), quiet=True)
        v = mmcv.VideoReader(out_file)
        assert v.resolution == (200, 100)
        os.remove(out_file)
        mmcv.resize_video(self.video_path, out_file, ratio=2)
        v = mmcv.VideoReader(out_file)
        assert v.resolution == (294 * 2, 240 * 2)
        os.remove(out_file)
        mmcv.resize_video(self.video_path, out_file, (1000, 480), keep_ar=True)
        v = mmcv.VideoReader(out_file)
        assert v.resolution == (294 * 2, 240 * 2)
        os.remove(out_file)
        mmcv.resize_video(
            self.video_path, out_file, ratio=(2, 1.5), keep_ar=True)
        v = mmcv.VideoReader(out_file)
        assert v.resolution == (294 * 2, 360)
        os.remove(out_file)
