# Copyright (c) OpenMMLab. All rights reserved.
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import mmcv
from mmcv.image.geometric import _scale_size
from .base import BaseTransform
from .builder import TRANSFORMS


@TRANSFORMS.register_module()
class Normalize(BaseTransform):
    """Normalize the image.

    Required Keys:

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
            should be BGR. Defaults to True.
    """

    def __init__(self,
                 mean: Sequence[float],
                 std: Sequence[float],
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
    - gt_semantic_seg (optional)
    - gt_keypoints (optional)

    Modified Keys:

    - img
    - gt_bboxes
    - gt_semantic_seg
    - gt_keypoints
    - height
    - width

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
        clip_object_border (bool, optional): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the gt
            bboxes are allowed to cross the border of images. Therefore, we
            don't need to clip the gt bboxes in these cases. Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend. Defaults
            to 'cv2'.
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
            results['height'], results['width'] = img.shape[:2]
            results['scale'] = img.shape[:2][::-1]
            results['scale_factor'] = (w_scale, h_scale)
            results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results: dict) -> None:
        """Resize bounding boxes with ``results['scale_factor']``."""
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes'] * np.tile(
                np.array(results['scale_factor']), 2)
            if self.clip_object_border:
                bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, results['width'])
                bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0,
                                          results['height'])
            results['gt_bboxes'] = bboxes

    def _resize_seg(self, results: dict) -> None:
        """Resize semantic segmentation map with ``results['scale']``."""
        if results.get('gt_semantic_seg', None) is not None:
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results['gt_semantic_seg'],
                    results['scale'],
                    interpolation=self.interpolation,
                    backend=self.backend)
            else:
                gt_seg = mmcv.imresize(
                    results['gt_semantic_seg'],
                    results['scale'],
                    interpolation=self.interpolation,
                    backend=self.backend)
            results['gt_semantic_seg'] = gt_seg

    def _resize_keypoints(self, results: dict) -> None:
        """Resize keypoints with ``results['scale_factor']``."""
        if results.get('gt_keypoints', None) is not None:
            keypoints = results['gt_keypoints']

            keypoints[:, :, :2] = keypoints[:, :, :2] * np.array(
                results['scale_factor'])
            if self.clip_object_border:
                keypoints[:, :, 0] = np.clip(keypoints[:, :, 0], 0,
                                             results['width'])
                keypoints[:, :, 1] = np.clip(keypoints[:, :, 1], 0,
                                             results['height'])
            results['gt_keypoints'] = keypoints

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_semantic_seg',
            'gt_keypoints', 'scale', 'scale_factor', 'height', 'width',
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
    - gt_semantic_seg (optional)

    Modified Keys:

    - img
    - gt_semantic_seg
    - height
    - width

    Added Keys:

    - pad_shape
    - pad_fixed_size
    - pad_size_divisor

    Args:
        size (tuple, optional): Fixed padding size.
            Expected padding shape (h, w)Defaults to None.
        size_divisor (int, optional): The divisor of padded size. Defaults to
            None.
        pad_to_square (bool): Whether to pad the image into a square.
            Currently only used for YOLOX. Defaults to False.
        pad_val (int or dict, optional): A dict for padding value.
            if ``type(pad_val) == int``, the val to pad seg is 255. Defaults to
            ``dict(img=0, seg=255)``.
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
                 pad_val: Union[int, dict] = dict(img=0, seg=255),
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
        padded_img = mmcv.impad(
            results['img'],
            shape=size,
            pad_val=pad_val,
            padding_mode=self.padding_mode)

        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor
        results['height'] = padded_img.shape[0]
        results['width'] = padded_img.shape[1]

    def _pad_seg(self, results: dict) -> None:
        """Pad semantic segmentation map according to
        ``results['pad_shape']``."""
        if results.get('gt_semantic_seg', None) is not None:
            pad_val = self.pad_val.get('seg', 255)

            results['gt_semantic_seg'] = mmcv.impad(
                results['gt_semantic_seg'],
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
class RandomFlip(BaseTransform):
    """Flip the image & bbox & keypoints & segmentation map.

    There are 3 flip modes:

    - ``prob`` is float, ``direction`` is string

        the image will be ``direction`` ly flipped with probability
        of ``prob`` . E.g., ``prob=0.5``, ``direction='horizontal'``,
        then image will be horizontally flipped with probability of 0.5.

    - ``prob`` is float, ``direction`` is list of string

        the image will be ``direction[i]`` ly flipped with probability of
        ``prob/len(direction)``. E.g., ``prob=0.5``,
        ``direction=['horizontal', 'vertical']``, then image will be
        horizontally flipped with probability of 0.25, vertically with
        probability of 0.25.

    - ``prob`` is list of float, ``direction`` is list of string

        given ``len(prob) == len(direction)``, the image will
        be ``direction[i]`` ly flipped with probability of ``prob[i]``.
        E.g., ``prob=[0.3, 0.5]``, ``direction=['horizontal',
        'vertical']``, then image will be horizontally flipped with
        probability of 0.3, vertically with probability of 0.5.

    Required Keys:

    - img
    - gt_bboxes
    - gt_semantic_seg
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_semantic_seg
    - gt_keypoints

    Added Keys:

    - flip
    - flip_direction

    Args:
         prob (float | list[float], optional): The flipping probability.
             Defaults to None.
         direction(str | list[str]): The flipping direction. Options
             If input is a list, the length must equal ``prob``. Each
             element in ``prob`` indicates the flip probability of
             corresponding direction. Defaults to horizontal.
    """

    def __init__(
            self,
            prob: Optional[Union[float, Iterable[float]]] = None,
            direction: Union[str,
                             Sequence[Optional[str]]] = 'horizontal') -> None:
        if isinstance(prob, list):
            assert mmcv.is_list_of(prob, float)
            assert 0 <= sum(prob) <= 1
        elif isinstance(prob, float):
            assert 0 <= prob <= 1
        else:
            raise ValueError(f"probs must be float or list of float, but \
                              got '{type(prob)}'.")
        self.prob = prob

        valid_directions = ['horizontal', 'vertical', 'diagonal']
        if isinstance(direction, str):
            assert direction in valid_directions
        elif isinstance(direction, list):
            assert mmcv.is_list_of(direction, str)
            assert set(direction).issubset(set(valid_directions))
        else:
            raise ValueError(f"direction must be either str or list of str, \
                               but got '{type(direction)}'.")
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
                  or 'diagnal', but got '{direction}'")
        return flipped

    def flip_keypoints(self, keypoints: np.ndarray, img_shape: Tuple[int, int],
                       direction: str) -> np.ndarray:
        """Flip keypoints horizontally, vertically or diagnally.

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
                  or 'diagnal', but got '{direction}'")
        flipped = np.concatenate([keypoints, meta_info], axis=-1)
        return flipped

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
        if results.get('gt_semantic_seg', None) is not None:
            results['gt_semantic_seg'] = mmcv.imflip(
                results['gt_semantic_seg'],
                direction=results['flip_direction'])

    def _flip_with_flip_direction(self, results: dict) -> None:
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
            dict: Flipped results, 'img', 'gt_bboxes', 'gt_semantic_seg',
            'gt_keypoints', 'flip', and 'flip_direction' keys are
            updated in result dict.
        """
        self._flip_with_flip_direction(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.prob}, '
        repr_str += f'interpolation={self.direction})'

        return repr_str


@TRANSFORMS.register_module()
class RandomResize(BaseTransform):
    """Random resize images & bbox & keypoints.

    Added or updated keys: scale, scale_factor, keep_ratio, img, height, width,
    gt_bboxes, gt_semantic_seg, and gt_keypoints.
    How to choose the target scale to resize the image will follow the rules
    below:

    - if `scale` is a list of tuple, the first value of the target scale is
      sampled from [`scale[0][0]`, `scale[1][0]`] uniformally and the second
      value of the target scale is sampled from [`scale[0][1]`, `scale[1][1]`]
      uniformally.
    - if `scale` is a tuple, the first and second values of the target scale
      is equal to the first and second values of `scale` multiplied by a value
      sampled from [`ratio_range[0]`, `ratio_range[1]`] uniformally.

    Required Keys:

    - img
    - gt_bboxes
    - gt_semantic_seg
    - gt_keypoints

    Modified Keys:

    - img
    - gt_bboxes
    - gt_semantic_seg
    - gt_keypoints

    Added Keys:

    - scale
    - scale_factor
    - keep_ratio

    Args:
        scale (tuple or list[tuple], optional): Images scales for resizing.
            Defaults to None.
        ratio_range (tuple[float], optional): (min_ratio, max_ratio).
            Defaults to None.
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image. Defaults to True.
        clip_object_border (bool): Whether to clip the objects
            outside the border of the image. In some dataset like MOT17, the
            gt bboxes are allowed to cross the border of images. Therefore,
            we don't need to clip the gt bboxes in these cases.
            Defaults to True.
        backend (str): Image resize backend, choices are 'cv2' and 'pillow'.
            These two backends generates slightly different results. Defaults
            to 'cv2'.
        interpolation (str): How to interpolate the original image when
            resizing. Defaults to 'bilinear'.
    """

    def __init__(self,
                 scale: Union[Tuple[int, int], List[Tuple[int, int]]] = None,
                 ratio_range: Tuple[float, float] = None,
                 keep_ratio: bool = True,
                 clip_object_border: bool = True,
                 backend: str = 'cv2',
                 interpolation: str = 'bilinear') -> None:

        assert scale is not None

        self.scale = scale
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.clip_object_border = clip_object_border
        self.backend = backend
        self.interpolation = interpolation

        # create a empty Reisize object
        self.resize = Resize(0)
        self.resize.keep_ratio = keep_ratio
        self.resize.clip_object_border = clip_object_border
        self.resize.backend = backend
        self.resize.interpolation = interpolation

    @staticmethod
    def _random_sample(scales: Sequence[Tuple[int, int]]) -> Tuple[int, int]:
        """Private function to randomly sample a scale from a list of tuples.

        Args:
            scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in scales, which specify the lower
                and upper bound of image scales.
        Returns:
            tuple: Returns the target scale.
        """

        assert mmcv.is_list_of(scales, tuple) and len(scales) == 2
        scale_long = [max(s) for s in scales]
        scale_short = [min(s) for s in scales]
        long_edge = np.random.randint(min(scale_long), max(scale_long) + 1)
        short_edge = np.random.randint(min(scale_short), max(scale_short) + 1)
        scale = (long_edge, short_edge)
        return scale

    @staticmethod
    def _random_sample_ratio(
            scale: tuple, ratio_range: Tuple[float, float]) -> Tuple[int, int]:
        """Private function to randomly sample a scale from a tuple.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``scale`` to
        generate sampled scale.
        Args:
            scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``scale``.
        Returns:
            tuple: Returns the target scale.
        """

        assert isinstance(scale, tuple) and len(scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(scale[0] * ratio), int(scale[1] * ratio)
        return scale

    def _random_scale(self, results: dict) -> None:
        """Private function to randomly sample an scale according to the type
        of `scale`.

        Args:
            results (dict): Result dict from :obj:`dataset`.
        Returns:
            dict: One new key 'scale`is added into ``results``,
            which would be used by subsequent pipelines.
        """

        if isinstance(self.scale, tuple):
            assert self.ratio_range is not None and len(self.ratio_range) == 2
            scale: Tuple[int, int] = self._random_sample_ratio(
                self.scale, self.ratio_range)
        elif mmcv.is_list_of(self.scale, tuple):
            scale = self._random_sample(self.scale)
        else:
            raise NotImplementedError(f"Do not support sampling function \
                                        for '{self.scale}'")

        results['scale'] = scale

    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_semantic_seg',
            'gt_keypoints', 'scale', 'scale_factor', 'height', 'width',
            and 'keep_ratio' keys are updated in result dict.
        """
        self._random_scale(results)
        self.resize.scale = results['scale']
        results = self.resize.transform(results)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(scale={self.scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'keep_ratio={self.keep_ratio}, '
        repr_str += f'bbox_clip_border={self.clip_object_border}, '
        repr_str += f'backend={self.backend}, '
        repr_str += f'interpolation={self.interpolation})'
        return repr_str
