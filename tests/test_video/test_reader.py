# Copyright (c) Open-MMLab. All rights reserved.
import os
import os.path as osp
import shutil
import tempfile
from collections import OrderedDict

import pytest

import mmcv


class TestCache:

    def test_init(self):
        with pytest.raises(ValueError):
            mmcv.Cache(0)
        cache = mmcv.Cache(100)
        assert cache.capacity == 100
        assert cache.size == 0

    def test_put(self):
        cache = mmcv.Cache(3)
        for i in range(1, 4):
            cache.put(f'k{i}', i)
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


class TestVideoReader:

    @classmethod
    def setup_class(cls):
        cls.video_path = osp.join(osp.dirname(__file__), '../data/test.mp4')
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
            filename = f'{frame_dir}/{i:06d}.jpg'
            assert osp.isfile(filename)
            os.remove(filename)

        v = mmcv.VideoReader(self.video_path)
        v.cvt2frames(frame_dir, show_progress=False)
        assert osp.isdir(frame_dir)
        for i in range(self.num_frames):
            filename = f'{frame_dir}/{i:06d}.jpg'
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
            filename = f'{frame_dir}/{i:03d}.JPEG'
            assert osp.isfile(filename)
            os.remove(filename)
        shutil.rmtree(frame_dir)

    def test_frames2video(self):
        v = mmcv.VideoReader(self.video_path)
        frame_dir = tempfile.mkdtemp()
        v.cvt2frames(frame_dir)
        assert osp.isdir(frame_dir)
        for i in range(self.num_frames):
            filename = f'{frame_dir}/{i:06d}.jpg'
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
            filename = f'{frame_dir}/{i:06d}.jpg'
            os.remove(filename)
        shutil.rmtree(frame_dir)
        os.remove(out_filename)
