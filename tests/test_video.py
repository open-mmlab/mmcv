import os
import os.path as osp
import tempfile
from collections import OrderedDict

import mmcv
import pytest


class TestCache(object):

    def test_init(self):
        with pytest.raises(ValueError):
            mmcv.Cache(0)
        cache = mmcv.Cache(100)
        assert cache.capacity == 100
        assert cache.size == 0

    def test_put(self):
        cache = mmcv.Cache(3)
        for i in range(1, 4):
            cache.put('k{}'.format(i), i)
            assert cache.size == i
        assert cache._cache == OrderedDict([('k1', 1), ('k2', 2), ('k3', 3)])
        cache.put('k4', 4)
        assert cache.size == 3
        assert cache._cache == OrderedDict([('k2', 2), ('k3', 3), ('k4', 4)])
        cache.put('k2', 2)
        assert cache._cache == OrderedDict([('k2', 2), ('k3', 3), ('k4', 4)])

    def test_get(self):
        cache = mmcv.Cache(3)
        assert cache.get('key_none') is None
        assert cache.get('key_none', 0) == 0
        cache.put('k1', 1)
        assert cache.get('k1') == 1


class TestVideo(object):

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), 'data/test.mp4')
        cls.num_frames = 168

    def test_load(self):
        v = mmcv.VideoReader(self.video_path)
        assert v.width == 294
        assert v.height == 240
        assert v.fps == 25
        assert v.frame_cnt == self.num_frames
        assert len(v) == self.num_frames
        assert v.opened
        import cv2
        assert isinstance(v.vcap, type(cv2.VideoCapture()))

    def test_read(self):
        v = mmcv.VideoReader(self.video_path)
        img = v.read()
        assert int(round(img.mean())) == 94
        img = v.get_frame(63)
        assert int(round(img.mean())) == 94
        img = v[64]
        assert int(round(img.mean())) == 205
        img = v[-104]
        assert int(round(img.mean())) == 205
        img = v[63]
        assert int(round(img.mean())) == 94
        img = v[-105]
        assert int(round(img.mean())) == 94
        img = v.read()
        assert int(round(img.mean())) == 205
        with pytest.raises(IndexError):
            v.get_frame(self.num_frames + 1)
        with pytest.raises(IndexError):
            v[-self.num_frames - 1]

    def test_slice(self):
        v = mmcv.VideoReader(self.video_path)
        imgs = v[-105:-103]
        assert int(round(imgs[0].mean())) == 94
        assert int(round(imgs[1].mean())) == 205
        assert len(imgs) == 2
        imgs = v[63:65]
        assert int(round(imgs[0].mean())) == 94
        assert int(round(imgs[1].mean())) == 205
        assert len(imgs) == 2
        imgs = v[64:62:-1]
        assert int(round(imgs[0].mean())) == 205
        assert int(round(imgs[1].mean())) == 94
        assert len(imgs) == 2
        imgs = v[:5]
        assert len(imgs) == 5
        for img in imgs:
            assert int(round(img.mean())) == 94
        imgs = v[165:]
        assert len(imgs) == 3
        for img in imgs:
            assert int(round(img.mean())) == 0
        imgs = v[-3:]
        assert len(imgs) == 3
        for img in imgs:
            assert int(round(img.mean())) == 0

    def test_current_frame(self):
        v = mmcv.VideoReader(self.video_path)
        assert v.current_frame() is None
        v.read()
        img = v.current_frame()
        assert int(round(img.mean())) == 94

    def test_position(self):
        v = mmcv.VideoReader(self.video_path)
        assert v.position == 0
        for _ in range(10):
            v.read()
        assert v.position == 10
        v.get_frame(99)
        assert v.position == 100

    def test_iterator(self):
        cnt = 0
        for img in mmcv.VideoReader(self.video_path):
            cnt += 1
            assert img.shape == (240, 294, 3)
        assert cnt == self.num_frames

    def test_with(self):
        with mmcv.VideoReader(self.video_path) as v:
            assert v.opened
        assert not v.opened

    def test_cvt2frames(self):
        v = mmcv.VideoReader(self.video_path)
        frame_dir = tempfile.mkdtemp()
        v.cvt2frames(frame_dir)
        assert osp.isdir(frame_dir)
        for i in range(self.num_frames):
            filename = '{}/{:06d}.jpg'.format(frame_dir, i)
            assert osp.isfile(filename)
            os.remove(filename)

        v = mmcv.VideoReader(self.video_path)
        v.cvt2frames(frame_dir, show_progress=False)
        assert osp.isdir(frame_dir)
        for i in range(self.num_frames):
            filename = '{}/{:06d}.jpg'.format(frame_dir, i)
            assert osp.isfile(filename)
            os.remove(filename)

        v = mmcv.VideoReader(self.video_path)
        v.cvt2frames(
            frame_dir,
            file_start=100,
            filename_tmpl='{:03d}.JPEG',
            start=100,
            max_num=20)
        assert osp.isdir(frame_dir)
        for i in range(100, 120):
            filename = '{}/{:03d}.JPEG'.format(frame_dir, i)
            assert osp.isfile(filename)
            os.remove(filename)
        os.removedirs(frame_dir)

    def test_frames2video(self):
        v = mmcv.VideoReader(self.video_path)
        frame_dir = tempfile.mkdtemp()
        v.cvt2frames(frame_dir)
        assert osp.isdir(frame_dir)
        for i in range(self.num_frames):
            filename = '{}/{:06d}.jpg'.format(frame_dir, i)
            assert osp.isfile(filename)

        out_filename = osp.join(tempfile.gettempdir(), 'mmcv_test.avi')
        mmcv.frames2video(frame_dir, out_filename)
        v = mmcv.VideoReader(out_filename)
        assert v.fps == 30
        assert len(v) == self.num_frames

        mmcv.frames2video(
            frame_dir,
            out_filename,
            fps=25,
            start=10,
            end=50,
            show_progress=False)
        v = mmcv.VideoReader(out_filename)
        assert v.fps == 25
        assert len(v) == 40

        for i in range(self.num_frames):
            filename = '{}/{:06d}.jpg'.format(frame_dir, i)
            os.remove(filename)
        os.removedirs(frame_dir)
        os.remove(out_filename)

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
