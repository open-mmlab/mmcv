# Copyright (c) OpenMMLab. All rights reserved.
import copy
import random
import warnings
from itertools import product
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np

import mmcv
from mmcv.image.geometric import _scale_size
from .base import BaseTransform
from .builder import TRANSFORMS
from .utils import cache_randomness
from .wrappers import Compose

Number = Union[int, float]


@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    """Normalize the image.

    Required Keys:

    - img

    Modified Keys:

    - img

    Added Keys:

    - img_norm_cfg

      - mean
      - std
      - to_rgb


    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB before
            normlizing the image. If ``to_rgb=True``, the order of mean and std
            should be RGB. If ``to_rgb=False``, the order of mean and std
            should be the same order of the image. Defaults to True.
    """

    def __init__(self,
                 mean: Sequence[Number],
                 std: Sequence[Number],
                 to_rgb: bool = True) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def transform(self, results: dict) -> dict:
        """Function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, key 'img_norm_cfg' key is added in to
            result dict.
        """

        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


@TRANSFORMS.register_module()
class Resize(BaseTransform):
    """Resize images & bbox & seg & keypoints.

    This transform resizes the input image according to ``scale`` or
    ``scale_factor``. Bboxes, seg map and keypoints are then resized with the
    same scale factor.
    if ``scale`` and ``scale_factor`` are both set, it will use ``scale`` to
    resize.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (int or tuple): Images scales for resizing. Defaults to None
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to False.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'bilinear'.
    """

    def __init__(self,
                 scale: Optional[Union[int, Tuple[int, int]]] = None,
                 scale_factor: Optional[Union[float, Tuple[float,
                                                           float]]] = None,
                 keep_ratio: bool = False,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation='bilinear') -> None:
        assert scale is not None or scale_factor is not None, (
            '`scale` and'
            '`scale_factor` can not both be `None`')
        if scale is None:
            self.scale = None
        else:
            if isinstance(scale, int):
                self.scale = (scale, scale)
            else:
                self.scale = scale

        self.backend = backend
        self.interpolation = interpolation
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        if scale_factor is None:
            self.scale_factor = None
        elif isinstance(scale_factor, float):
            self.scale_factor = (scale_factor, scale_factor)
        elif isinstance(scale_factor, tuple):
            assert (len(scale_factor)) == 2
            self.scale_factor = scale_factor
        else:
            raise TypeError(
                f'expect scale_factor is float or Tuple(float), but'
                f'get {type(scale_factor)}')

    def _resize_img(self, results: dict) -> None:
        """Resize images with ``results['scale']``."""

        if results.get('img', None) is not None:
            if self.keep_ratio:
                img, scale_factor = mmcv.imrescale(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
                # the w_scale and h_scale has minor difference
                # a real fix should be done in the mmcv.imrescale in the future
                new_h, new_w = img.shape[:2]
                h, w = results['img'].shape[:2]
                w_scale = new_w / w
                h_scale = new_h / h
            else:
                img, w_scale, h_scale = mmcv.imresize(
                    results['img'],
                    results['scale'],
                    interpolation=self.interpolation,
                    return_scale=True,
                    backend=self.backend)
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0,
                                          results['img_shape'][1])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['img_shape'][0])
            results['gt_bboxes'] = bboxes

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_seg_map', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_seg_map'],
                    results['scale'],
                    interpolation='nearest',
                    backend=self.backend)
            results['gt_seg_map'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['img_shape'][1])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['img_shape'][0])
            results['gt_keypoints'] = keypoints

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = _scale_size(img_shape[::-1], self.scale_factor)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'scale_factor={self.scale_factor}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'clip_object_border={self.clip_object_border}), '
        repr_str += f'backend={self.backend}), '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str


@TRANSFORMS.register_module()
class Pad(BaseTransform):
    """Pad the image & segmentation map.

    There are three padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number. and (3)pad to square. Also,
    pad to square and pad to the minimum size can be used as the same time.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)

    Modified Keys:

    - img
    - gt_seg_map
    - img_shape

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (w, h). Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (Number | dict[str, Number], optional) - Padding value for if
            the pad_mode is "constant".  If it is a single number, the value
            to pad the image is the number and to pad the semantic
            segmentation map is 255. If it is a dict, it should have the
            following keys:

            - img: The value to pad the image.
            - seg: The value to pad the semantic segmentation map.
            Defaults to dict(img=0, seg=255).
        padding_mode (str): Type of padding. Should be: constant, edge,
            reflect or symmetric. Defaults to 'constant'.

            - constant: pads with a constant value, this value is specified
              with pad_val.
            - edge: pads with the last value at the edge of the image.
            - reflect: pads with reflection of image without repeating the last
              value on the edge. For example, padding [1, 2, 3, 4] with 2
              elements on both sides in reflect mode will result in
              [3, 2, 1, 2, 3, 4, 3, 2].
            - symmetric: pads with reflection of image repeating the last value
              on the edge. For example, padding [1, 2, 3, 4] with 2 elements on
              both sides in symmetric mode will result in
              [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self,
                 size: Optional[Tuple[int, int]] = None,
                 size_divisor: Optional[int] = None,
                 pad_to_square: bool = False,
                 pad_val: Union[Number, dict] = dict(img=0, seg=255),
                 padding_mode: str = 'constant') -> None:
        self.size = size
        self.size_divisor = size_divisor
        if isinstance(pad_val, int):
            pad_val = dict(img=pad_val, seg=255)
        assert isinstance(pad_val, dict), 'pad_val '
        self.pad_val = pad_val
        self.pad_to_square = pad_to_square

        if pad_to_square:
            assert size is None, \
                'The size and size_divisor must be None ' \
                'when pad2square is True'
        else:
            assert size is not None or size_divisor is not None, \
                'only one of size and size_divisor should be valid'
            assert size is None or size_divisor is None
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode

    def _pad_img(self, results: dict) -> None:
        """Pad images according to ``self.size``."""
        pad_val = self.pad_val.get('img', 0)

        size = None
        if self.pad_to_square:
            max_size = max(results['img'].shape[:2])
            size = (max_size, max_size)
        if self.size_divisor is not None:
            if size is None:
                size = (results['img'].shape[0], results['img'].shape[1])
            pad_h = int(np.ceil(
                size[0] / self.size_divisor)) * self.size_divisor
            pad_w = int(np.ceil(
                size[1] / self.size_divisor)) * self.size_divisor
            size = (pad_h, pad_w)
        elif self.size is not None:
            size = self.size[::-1]
        if isinstance(pad_val, int) and results['img'].ndim == 3:
            pad_val = tuple(pad_val for _ in range(results['img'].shape[2]))
        padded_img = mmcv.impad(
            results['img'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['img_shape'] = padded_img.shape[:2]

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        if results.get('gt_seg_map', None) is not None:
            pad_val = self.pad_val.get('seg', 255)
            if isinstance(pad_val, int) and results['gt_seg_map'].ndim == 3:
                pad_val = tuple(
                    pad_val for _ in range(results['gt_seg_map'].shape[2]))
            results['gt_seg_map'] = mmcv.impad(
                results['gt_seg_map'],
                shape=results['pad_shape'][:2],
                pad_val=pad_val,
                padding_mode=self.padding_mode)

    def transform(self, results: dict) -> dict:
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        repr_str += f'size_divisor={self.size_divisor}, '
        repr_str += f'pad_to_square={self.pad_to_square}, '
        repr_str += f'pad_val={self.pad_val}), '
        repr_str += f'padding_mode={self.padding_mode})'
        return repr_str


@TRANSFORMS.register_module()
class CenterCrop(BaseTransform):
    """Crop the center of the image, segmentation masks, bounding boxes and key
    points. If the crop area exceeds the original image and ``auto_pad`` is
    True, the original image will be padded before cropping.

    Required Keys:

    - img
    - gt_seg_map (optional)
    - gt_bboxes (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_seg_map (optional)
    - gt_bboxes (optional)
    - gt_keypoints (optional)

    Added Key:

    - pad_shape


    Args:
        crop_size (Union[int, Tuple[int, int]]):  Expected size after cropping
            with the format of (w, h). If set to an integer, then cropping
            width and height are equal to this integer.
        auto_pad (bool): Whether to pad the image if it's smaller than the
            ``crop_size``. Defaults to False.
        pad_cfg (dict): Base config for padding. Refer to ``mmcv.Pad`` for
            detail. Defaults to ``dict(type='Pad')``.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the
            gt bboxes are allowed to cross the border of images. Therefore,
            we don't need to clip the gt bboxes in these cases.
            Defaults to True.
    """

    def __init__(self,
                 crop_size: Union[int, Tuple[int, int]],
                 auto_pad: bool = False,
                 pad_cfg: dict = dict(type='Pad'),
                 clip_object_border: bool = True) -> None:
        super().__init__()
        assert isinstance(crop_size, int) or (
            isinstance(crop_size, tuple) and len(crop_size) == 2
        ), 'The expected crop_size is an integer, or a tuple containing two '
        'intergers'

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.auto_pad = auto_pad

        self.pad_cfg = pad_cfg.copy()
        # size will be overwritten
        if 'size' in self.pad_cfg and auto_pad:
            warnings.warn('``size`` is set in ``pad_cfg``,'
                          'however this argument will be overwritten'
                          ' according to crop size and image size')

        self.clip_object_border = clip_object_border

    def _crop_img(self, results: dict, bboxes: np.ndarray) -> None:
        """Crop image.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if results.get('img', None) is not None:
            img = mmcv.imcrop(results['img'], bboxes=bboxes)
            img_shape = img.shape[:2]
            results['img'] = img
            results['img_shape'] = img_shape
            results['pad_shape'] = img_shape

    def _crop_seg_map(self, results: dict, bboxes: np.ndarray) -> None:
        """Crop semantic segmentation map.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if results.get('gt_seg_map', None) is not None:
            img = mmcv.imcrop(results['gt_seg_map'], bboxes=bboxes)
            results['gt_seg_map'] = img

    def _crop_bboxes(self, results: dict, bboxes: np.ndarray) -> None:
        """Update bounding boxes according to CenterCrop.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if 'gt_bboxes' in results:
            offset_w = bboxes[0]
            offset_h = bboxes[1]
            bbox_offset = np.array([offset_w, offset_h, offset_w, offset_h])
            # gt_bboxes has shape (num_gts, 4) in (tl_x, tl_y, br_x, br_y)
            # order.
            gt_bboxes = results['gt_bboxes'] - bbox_offset
            if self.clip_object_border:
                gt_bboxes[:, 0::2] = np.clip(gt_bboxes[:, 0::2], 0,
                                             results['img'].shape[1])
                gt_bboxes[:, 1::2] = np.clip(gt_bboxes[:, 1::2], 0,
                                             results['img'].shape[0])
            results['gt_bboxes'] = gt_bboxes

    def _crop_keypoints(self, results: dict, bboxes: np.ndarray) -> None:
        """Update key points according to CenterCrop. Keypoints that not in the
        cropped image will be set invisible.

        Args:
            results (dict): Result dict contains the data to transform.
            bboxes (np.ndarray): Shape (4, ), location of cropped bboxes.
        """
        if 'gt_keypoints' in results:
            offset_w = bboxes[0]
            offset_h = bboxes[1]
            keypoints_offset = np.array([offset_w, offset_h, 0])
            # gt_keypoints has shape (N, NK, 3) in (x, y, visibility) order,
            # NK = number of points per object
            gt_keypoints = results['gt_keypoints'] - keypoints_offset
            # set gt_kepoints out of the result image invisible
            height, width = results['img'].shape[:2]
            valid_pos = (gt_keypoints[:, :, 0] >=
                         0) * (gt_keypoints[:, :, 0] <
                               width) * (gt_keypoints[:, :, 1] >= 0) * (
                                   gt_keypoints[:, :, 1] < height)
            gt_keypoints[:, :, 2] = np.where(valid_pos, gt_keypoints[:, :, 2],
                                             0)
            gt_keypoints[:, :, 0] = np.clip(gt_keypoints[:, :, 0], 0,
                                            results['img'].shape[1])
            gt_keypoints[:, :, 1] = np.clip(gt_keypoints[:, :, 1], 0,
                                            results['img'].shape[0])
            results['gt_keypoints'] = gt_keypoints

    def transform(self, results: dict) -> dict:
        """Apply center crop on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Results with CenterCropped image and semantic segmentation
            map.
        """
        crop_width, crop_height = self.crop_size[0], self.crop_size[1]

        assert 'img' in results, '`img` is not found in results'
        img = results['img']
        # img.shape has length 2 for grayscale, length 3 for color
        img_height, img_width = img.shape[:2]

        if crop_height > img_height or crop_width > img_width:
            if self.auto_pad:
                # pad the area
                img_height = max(img_height, crop_height)
                img_width = max(img_width, crop_width)
                pad_size = (img_width, img_height)
                _pad_cfg = self.pad_cfg.copy()
                _pad_cfg.update(dict(size=pad_size))
                pad_transform = TRANSFORMS.build(_pad_cfg)
                results = pad_transform(results)
            else:
                crop_height = min(crop_height, img_height)
                crop_width = min(crop_width, img_width)

        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])

        # crop the image
        self._crop_img(results, bboxes)
        # crop the gt_seg_map
        self._crop_seg_map(results, bboxes)
        # crop the bounding box
        self._crop_bboxes(results, bboxes)
        # crop the keypoints
        self._crop_keypoints(results, bboxes)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size = {self.crop_size}'
        repr_str += f', auto_pad={self.auto_pad}'
        repr_str += f', pad_cfg={self.pad_cfg}'
        repr_str += f',clip_object_border = {self.clip_object_border})'
        return repr_str


@TRANSFORMS.register_module()
class RandomGrayscale(BaseTransform):
    """Randomly convert image to grayscale with a probability.

    Required Key:

    - img

    Modified Key:

    - img

    Added Keys:

    - grayscale
    - grayscale_weights

    Args:
        prob (float): Probability that image should be converted to
            grayscale. Defaults to 0.1.
        keep_channels (bool): Whether keep channel number the same as
            input. Defaults to False.
        channel_weights (tuple): The grayscale weights of each channel,
            and the weights will be normalized. For example, (1, 2, 1)
            will be normalized as (0.25, 0.5, 0.25). Defaults to
            (1., 1., 1.).
        color_format (str): Color format set to be any of 'bgr',
            'rgb', 'hsv'. Note: 'hsv' image will be transformed into 'bgr'
            format no matter whether it is grayscaled. Defaults to 'bgr'.
    """

    def __init__(self,
                 prob: float = 0.1,
                 keep_channels: bool = False,
                 channel_weights: Sequence[float] = (1., 1., 1.),
                 color_format: str = 'bgr') -> None:
        super().__init__()
        assert 0. <= prob <= 1., ('The range of ``prob`` value is [0., 1.],' +
                                  f' but got {prob} instead')
        self.prob = prob
        self.keep_channels = keep_channels
        self.channel_weights = channel_weights
        assert color_format in ['bgr', 'rgb', 'hsv']
        self.color_format = color_format

    @cache_randomness
    def _random_prob(self):
        return random.random()

    def transform(self, results: dict) -> dict:
        """Apply random grayscale on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
           dict: Results with grayscale image.
        """
        img = results['img']
        # convert hsv to bgr
        if self.color_format == 'hsv':
            img = mmcv.hsv2bgr(img)
        img = img[..., None] if img.ndim == 2 else img
        num_output_channels = img.shape[2]
        if self._random_prob() < self.prob:
            if num_output_channels > 1:
                assert num_output_channels == len(
                    self.channel_weights
                ), 'The length of ``channel_weights`` are supposed to be '
                f'num_output_channels, but got {len(self.channel_weights)}'
                ' instead.'
                normalized_weights = (
                    np.array(self.channel_weights) / sum(self.channel_weights))
                img = (normalized_weights * img).sum(axis=2)
                img = img.astype('uint8')
                if self.keep_channels:
                    img = img[:, :, None]
                    results['img'] = np.dstack(
                        [img for _ in range(num_output_channels)])
                else:
                    results['img'] = img
                return results
        img = img.astype('uint8')
        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob = {self.prob}'
        repr_str += f', keep_channels = {self.keep_channels}'
        repr_str += f', channel_weights = {self.channel_weights}'
        repr_str += f', color_format = {self.color_format})'
        return repr_str


@TRANSFORMS.register_module()
class MultiScaleFlipAug(BaseTransform):
    """Test-time augmentation with multiple scales and flipping.

    An example configuration is as followed:

    .. code-block::

        dict(
            type='MultiScaleFlipAug',
            scales=[(1333, 400), (1333, 800)],
            flip=True,
            transforms=[
                dict(type='Normalize', **img_norm_cfg),
                dict(type='Pad', size_divisor=1),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])

    ``results`` will be resized using all the sizes in ``scales``.
    If ``flip`` is True, then flipped results will also be added into output
    list.

    For the above configuration, there are four combinations of resize
    and flip:

    - Resize to (1333, 400) + no flip
    - Resize to (1333, 400) + flip
    - Resize to (1333, 800) + no flip
    - resize to (1333, 800) + flip

    The four results are then transformed with ``transforms`` argument.
    After that, results are wrapped into lists of the same length as below:

    .. code-block::

        dict(
            inputs=[...],
            data_samples=[...]
        )

    Where the length of ``inputs`` and ``data_samples`` are both 4.

    Required Keys:

    - Depending on the requirements of the ``transforms`` parameter.

    Modified Keys:

    - All output keys of each transform.

    Args:
        transforms (list[dict]): Transforms to be applied to each resized
            and flipped data.
        scales (tuple | list[tuple] | None): Images scales for resizing.
        scale_factor (float or tuple[float]): Scale factors for resizing.
            Defaults to None.
        allow_flip (bool): Whether apply flip augmentation. Defaults to False.
        flip_direction (str | list[str]): Flip augmentation directions,
            options are "horizontal", "vertical" and "diagonal". If
            flip_direction is a list, multiple flip augmentations will be
            applied. It has no effect when flip == False. Defaults to
            "horizontal".
        resize_cfg (dict): Base config for resizing. Defaults to
            ``dict(type='Resize', keep_ratio=True)``.
        flip_cfg (dict): Base config for flipping. Defaults to
            ``dict(type='RandomFlip')``.
    """

    def __init__(
        self,
        transforms: List[dict],
        scales: Optional[Union[Tuple, List[Tuple]]] = None,
        scale_factor: Optional[Union[float, List[float]]] = None,
        allow_flip: bool = False,
        flip_direction: Union[str, List[str]] = 'horizontal',
        resize_cfg: dict = dict(type='Resize', keep_ratio=True),
        flip_cfg: dict = dict(type='RandomFlip')
    ) -> None:
        super().__init__()
        self.transforms = Compose(transforms)  # type: ignore

        if scales is not None:
            self.scales = scales if isinstance(scales, list) else [scales]
            self.scale_key = 'scale'
            assert mmengine.is_list_of(self.scales, tuple)
        else:
            # if ``scales`` and ``scale_factor`` both be ``None``
            if scale_factor is None:
                self.scales = [1.]  # type: ignore
            elif isinstance(scale_factor, list):
                self.scales = scale_factor  # type: ignore
            else:
                self.scales = [scale_factor]  # type: ignore

            self.scale_key = 'scale_factor'

        self.allow_flip = allow_flip
        self.flip_direction = flip_direction if isinstance(
            flip_direction, list) else [flip_direction]
        assert mmengine.is_list_of(self.flip_direction, str)
        if not self.allow_flip and self.flip_direction != ['horizontal']:
            warnings.warn(
                'flip_direction has no effect when flip is set to False')
        self.resize_cfg = resize_cfg.copy()
        self.flip_cfg = flip_cfg

    def transform(self, results: dict) -> Dict:
        """Apply test time augment transforms on results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The augmented data, where each value is wrapped
            into a list.
        """

        data_samples = []
        inputs = []
        flip_args = [(False, '')]
        if self.allow_flip:
            flip_args += [(True, direction)
                          for direction in self.flip_direction]
        for scale in self.scales:
            for flip, direction in flip_args:
                _resize_cfg = self.resize_cfg.copy()
                _resize_cfg.update({self.scale_key: scale})
                _resize_flip = [_resize_cfg]

                if flip:
                    _flip_cfg = self.flip_cfg.copy()
                    _flip_cfg.update(prob=1.0, direction=direction)
                    _resize_flip.append(_flip_cfg)
                else:
                    results['flip'] = False
                    results['flip_direction'] = None

                resize_flip = Compose(_resize_flip)
                _results = resize_flip(results.copy())
                packed_results = self.transforms(_results)  # type: ignore

                inputs.append(packed_results['inputs'])  # type: ignore
                data_samples.append(
                    packed_results['data_sample'])  # type: ignore
        return dict(inputs=inputs, data_sample=data_samples)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(transforms={self.transforms}'
        repr_str += f', scales={self.scales}'
        repr_str += f', allow_flip={self.allow_flip}'
        repr_str += f', flip_direction={self.flip_direction})'
        return repr_str


@TRANSFORMS.register_module()
class TestTimeAug(BaseTransform):
    """Test-time augmentation transform.

    An example configuration is as followed:

    .. code-block::

        dict(type='TestTimeAug',
             transforms=[
                [dict(type='Resize', scale=(1333, 800), keep_ratio=True),
                 dict(type='Resize', scale=(1333, 800), keep_ratio=True)],
                [dict(type='RandomFlip', prob=1.),
                 dict(type='RandomFlip', prob=0.)],
                [dict(type='PackDetInputs',
                      meta_keys=('img_id', 'img_path', 'ori_shape',
                                 'img_shape', 'scale_factor', 'flip',
                                 'flip_direction'))]])

    ``results`` will be transformed using all transforms defined in
    ``transforms`` arguments.

    For the above configuration, there are four combinations of resize
    and flip:

    - Resize to (1333, 400) + no flip
    - Resize to (1333, 400) + flip
    - Resize to (1333, 800) + no flip
    - resize to (1333, 800) + flip

    After that, results are wrapped into lists of the same length as followed:

    .. code-block::

        dict(
            inputs=[...],
            data_samples=[...]
        )

    The length of ``inputs`` and ``data_samples`` are both 4.

    Required Keys:

    - Depending on the requirements of the ``transforms`` parameter.

    Modified Keys:

    - All output keys of each transform.

    Args:
        transforms (list[list[dict]]): Transforms to be applied to data sampled
            from dataset. ``transforms`` is a list of list, and each list
            element usually represents a series of transforms with the same
            type and different arguments. Data will be processed by each list
            elements sequentially. See more information in :meth:`transform`.
    """

    def __init__(self, transforms: list):
        for i, transform_list in enumerate(transforms):
            for j, transform in enumerate(transform_list):
                if isinstance(transform, dict):
                    transform_list[j] = TRANSFORMS.build(transform)
                elif callable(transform):
                    continue
                else:
                    raise TypeError(
                        'transform must be callable or a dict, but got'
                        f' {type(transform)}')
            transforms[i] = transform_list

        self.subroutines = [
            Compose(subroutine) for subroutine in product(*transforms)
        ]

    def transform(self, results: dict) -> dict:
        """Apply all transforms defined in :attr:`transforms` to the results.

        As the example given in :obj:`TestTimeAug`, ``transforms`` consists of
        2 ``Resize``, 2 ``RandomFlip`` and 1 ``PackDetInputs``.
        The data sampled from dataset will be processed as follows:

        1. Data will be processed by 2 ``Resize`` and return a list
           of 2 results.
        2. Each result in list will be further passed to 2
           ``RandomFlip``, and aggregates into a list of 4 results.
        3. Each result will be processed by ``PackDetInputs``, and
           return a list of dict.
        4. Aggregates the same fields of results, and finally return
           a dict. Each value of the dict represents 4 transformed
           results.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: The augmented data, where each value is wrapped
            into a list.
        """
        results_list = []  # type: ignore
        for subroutine in self.subroutines:
            result = subroutine(copy.deepcopy(results))
            assert isinstance(result, dict), (
                f'Data processed by {subroutine} must return a dict, but got '
                f'{result}')
            assert result is not None, (
                f'Data processed by {subroutine} in `TestTimeAug` should not '
                'be None! Please check your validation dataset and the '
                f'transforms in {subroutine}')
            results_list.append(result)

        aug_data_dict = {
            key: [item[key] for item in results_list]  # type: ignore
            for key in results_list[0]  # type: ignore
        }
        return aug_data_dict

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += 'transforms=\n'
        for subroutine in self.subroutines:
            repr_str += f'{repr(subroutine)}\n'
        return repr_str


@TRANSFORMS.register_module()
class RandomChoiceResize(BaseTransform):
    """Resize images & bbox & mask from a list of multiple scales.

    This transform resizes the input image to some scale. Bboxes and masks are
    then resized with the same scale factor. Resize scale will be randomly
    selected from ``scales``.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if `scale` is a list of tuple, the target scale is sampled from the list
      uniformally.
    - if `scale` is a tuple, the target scale will be set to the tuple.

    Required Keys:

    - img
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_seg_map (optional)
    - gt_keypoints (optional)

    Added Keys:

    - scale
    - scale_factor
    - scale_idx
    - keep_ratio


    Args:
        scales (Union[list, Tuple]): Images scales for resizing.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        scales: Sequence[Union[int, Tuple]],
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:
        super().__init__()
        if isinstance(scales, list):
            self.scales = scales
        else:
            self.scales = [scales]
        assert mmengine.is_list_of(self.scales, tuple)

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Resize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    @cache_randomness
    def _random_select(self) -> Tuple[int, int]:
        """Randomly select an scale from given candidates.

        Returns:
            (tuple, int): Returns a tuple ``(scale, scale_dix)``,
            where ``scale`` is the selected image scale and
            ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmengine.is_list_of(self.scales, tuple)
        scale_idx = np.random.randint(len(self.scales))
        scale = self.scales[scale_idx]
        return scale, scale_idx

    def transform(self, results: dict) -> dict:
        """Apply resize transforms on results from a list of scales.

        Args:
            results (dict): Result dict contains the data to transform.

        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """

        target_scale, scale_idx = self._random_select()
        self.resize.scale = target_scale
        results = self.resize(results)
        results['scale_idx'] = scale_idx
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scales={self.scales}'
        repr_str += f', resize_cfg={self.resize_cfg})'
        return repr_str


@TRANSFORMS.register_module()
class RandomFlip(BaseTransform):
    """Flip the image & bbox & keypoints & segmentation map. Added or Updated
    keys: flip, flip_direction, img, gt_bboxes, gt_seg_map, and
    gt_keypoints. There are 3 flip modes:

     - ``prob`` is float, ``direction`` is string: the image will be
         ``direction``ly flipped with probability of ``prob`` .
         E.g., ``prob=0.5``, ``direction='horizontal'``,
         then image will be horizontally flipped with probability of 0.5.
     - ``prob`` is float, ``direction`` is list of string: the image will
         be ``direction[i]``ly flipped with probability of
         ``prob/len(direction)``.
         E.g., ``prob=0.5``, ``direction=['horizontal', 'vertical']``,
         then image will be horizontally flipped with probability of 0.25,
         vertically with probability of 0.25.
     - ``prob`` is list of float, ``direction`` is list of string:
         given ``len(prob) == len(direction)``, the image will
         be ``direction[i]``ly flipped with probability of ``prob[i]``.
         E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
         'vertical']``, then image will be horizontally flipped with
         probability of 0.3, vertically with probability of 0.5.

    Required Keys:
        - img
        - gt_bboxes (optional)
        - gt_seg_map (optional)
        - gt_keypoints (optional)

    Modified Keys:
        - img
        - gt_bboxes (optional)
        - gt_seg_map (optional)
        - gt_keypoints (optional)

    Added Keys:
        - flip
        - flip_direction
    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to 'horizontal'.
    """

    def __init__(
            self,
            prob: Optional[Union[float, Iterable[float]]] = None,
            direction: Union[str,
                             Sequence[Optional[str]]] = 'horizontal') -> None:
        if isinstance(prob, list):
            assert mmengine.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f'probs must be float or list of float, but \
                              got `{type(prob)}`.')
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmengine.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f'direction must be either str or list of str, \
                               but got `{type(direction)}`.')
        self.direction = direction

        if isinstance(prob, list):
            assert len(prob) == len(self.direction)

    def flip_bbox(self, bboxes: np.ndarray, img_shape: Tuple[int, int],
                  direction: str) -> np.ndarray:
        """Flip bboxes horizontally.

        Args:
            bboxes (numpy.ndarray): Bounding boxes, shape (..., 4*k)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped bounding boxes.
        """
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
        elif direction == 'vertical':
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        elif direction == 'diagonal':
            flipped[..., 0::4] = w - bboxes[..., 2::4]
            flipped[..., 1::4] = h - bboxes[..., 3::4]
            flipped[..., 2::4] = w - bboxes[..., 0::4]
            flipped[..., 3::4] = h - bboxes[..., 1::4]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        return flipped

    def flip_keypoints(self, keypoints: np.ndarray, img_shape: Tuple[int, int],
                       direction: str) -> np.ndarray:
        """Flip keypoints horizontally, vertically or diagonally.

        Args:
            keypoints (numpy.ndarray): Keypoints, shape (..., 2)
            img_shape (tuple[int]): Image shape (height, width)
            direction (str): Flip direction. Options are 'horizontal',
                'vertical'.
        Returns:
            numpy.ndarray: Flipped keypoints.
        """

        meta_info = keypoints[..., 2:]
        keypoints = keypoints[..., :2]
        flipped = keypoints.copy()
        h, w = img_shape
        if direction == 'horizontal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
        elif direction == 'vertical':
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        elif direction == 'diagonal':
            flipped[..., 0::2] = w - keypoints[..., 0::2]
            flipped[..., 1::2] = h - keypoints[..., 1::2]
        else:
            raise ValueError(
                f"Flipping direction must be 'horizontal', 'vertical', \
                  or 'diagonal', but got '{direction}'")
        flipped = np.concatenate([keypoints, meta_info], axis=-1)
        return flipped

    @cache_randomness
    def _choose_direction(self) -> str:
        """Choose the flip direction according to `prob` and `direction`"""
        if isinstance(self.direction,
                      Sequence) and not isinstance(self.direction, str):
            # None means non-flip
            direction_list: list = list(self.direction) + [None]
        elif isinstance(self.direction, str):
            # None means non-flip
            direction_list = [self.direction, None]

        if isinstance(self.prob, list):
            non_prob: float = 1 - sum(self.prob)
            prob_list = self.prob + [non_prob]
        elif isinstance(self.prob, float):
            non_prob = 1. - self.prob
            # exclude non-flip
            single_ratio = self.prob / (len(direction_list) - 1)
            prob_list = [single_ratio] * (len(direction_list) - 1) + [non_prob]

        cur_dir = np.random.choice(direction_list, p=prob_list)

        return cur_dir

    def _flip(self, results: dict) -> None:
        """Flip images, bounding boxes, semantic segmentation map and
        keypoints."""
        # flip image
        results['img'] = mmcv.imflip(
            results['img'], direction=results['flip_direction'])

        img_shape = results['img'].shape[:2]

        # flip bboxes
        if results.get('gt_bboxes', None) is not None:
            results['gt_bboxes'] = self.flip_bbox(results['gt_bboxes'],
                                                  img_shape,
                                                  results['flip_direction'])

        # flip keypoints
        if results.get('gt_keypoints', None) is not None:
            results['gt_keypoints'] = self.flip_keypoints(
                results['gt_keypoints'], img_shape, results['flip_direction'])

        # flip segs
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = mmcv.imflip(
                results['gt_seg_map'], direction=results['flip_direction'])

    def _flip_on_direction(self, results: dict) -> None:
        """Function to flip images, bounding boxes, semantic segmentation map
        and keypoints."""
        cur_dir = self._choose_direction()
        if cur_dir is None:
            results['flip'] = False
            results['flip_direction'] = None
        else:
            results['flip'] = True
            results['flip_direction'] = cur_dir
            self._flip(results)

    def transform(self, results: dict) -> dict:
        """Transform function to flip images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        self._flip_on_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'direction={self.direction})'

        return repr_str


@TRANSFORMS.register_module()
class RandomResize(BaseTransform):
    """Random resize images & bbox & keypoints.

    How to choose the target scale to resize the image will follow the rules
    below:

    - if ``scale`` is a sequence of tuple

    .. math::
        target\\_scale[0] \\sim Uniform([scale[0][0], scale[1][0]])
    .. math::
        target\\_scale[1] \\sim Uniform([scale[0][1], scale[1][1]])

    Following the resize order of weight and height in cv2, ``scale[i][0]``
    is for width, and ``scale[i][1]`` is for height.

    - if ``scale`` is a tuple

    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[0]
    .. math::
        target\\_scale[0] \\sim Uniform([ratio\\_range[0], ratio\\_range[1]])
            * scale[1]

    Following the resize order of weight and height in cv2, ``ratio_range[0]``
    is for width, and ``ratio_range[1]`` is for height.

    - if ``keep_ratio`` is True, the minimum value of ``target_scale`` will be
    used to set the shorter side and the maximum value will be used to
    set the longer side.

    - if ``keep_ratio`` is False, the value of ``target_scale`` will be used to
    reisze the width and height accordingly.

    Required Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_seg_map
    - gt_keypoints
    - img_shape

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (tuple or Sequence[tuple]): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        resize_type (str): The type of resize class to use. Defaults to
            "Resize".
        **resize_kwargs: Other keyword arguments for the ``resize_type``.

    Note:
        By defaults, the ``resize_type`` is "Resize", if it's not overwritten
        by your registry, it indicates the :class:`mmcv.Resize`. And therefore,
        ``resize_kwargs`` accepts any keyword arguments of it, like
        ``keep_ratio``, ``interpolation`` and so on.

        If you want to use your custom resize class, the class should accept
        ``scale`` argument and have ``scale`` attribution which determines the
        resize shape.
    """

    def __init__(
        self,
        scale: Union[Tuple[int, int], Sequence[Tuple[int, int]]],
        ratio_range: Tuple[float, float] = None,
        resize_type: str = 'Resize',
        **resize_kwargs,
    ) -> None:

        self.scale = scale
        self.ratio_range = ratio_range

        self.resize_cfg = dict(type=resize_type, **resize_kwargs)
        # create a empty Reisize object
        self.resize = TRANSFORMS.build({'scale': 0, **self.resize_cfg})

    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> tuple:
        """Private function to randomly sample a scale from a list of tuples.

        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert mmengine.is_list_of(scales, tuple) and len(scales) == 2
        scale_0 = [scales[0][0], scales[1][0]]
        scale_1 = [scales[0][1], scales[1][1]]
        edge_0 = np.random.randint(min(scale_0), max(scale_0) + 1)
        edge_1 = np.random.randint(min(scale_1), max(scale_1) + 1)
        scale = (edge_0, edge_1)
        return scale

    @staticmethod
    def _random_sample_ratio(scale: tuple, ratio_range: Tuple[float,
                                                              float]) -> tuple:
        """Private function to randomly sample a scale from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.

        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    @cache_randomness
    def _random_scale(self) -> tuple:
        """Private function to randomly sample an scale according to the type
        of ``scale``.

        Returns:
            tuple: The targeted scale of the image to be resized.
        """

        if mmengine.is_tuple_of(self.scale, int):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale = self._random_sample_ratio(
                self.scale,  # type: ignore
                self.ratio_range)
        elif mmengine.is_seq_of(self.scale, tuple):
            scale = self._random_sample(self.scale)  # type: ignore
        else:
            raise NotImplementedError('Do not support sampling function '
                                      f'for "{self.scale}"')

        return scale

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, ``img``, ``gt_bboxes``, ``gt_semantic_seg``,
            ``gt_keypoints``, ``scale``, ``scale_factor``, ``img_shape``, and
            ``keep_ratio`` keys are updated in result dict.
        """
        results['scale'] = self._random_scale()
        self.resize.scale = results['scale']
        results = self.resize(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'resize_cfg={self.resize_cfg})'
        return repr_str
