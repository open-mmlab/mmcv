# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms.base import BaseTransform
from mmcv.transforms.builder import TRANSFORMS
from mmcv.transforms.loading import LoadAnnotations, LoadImageFromFile
from mmcv.transforms.processing import (
    CenterCrop,
    MultiScaleFlipAug,
    Normalize,
    Pad,
    RandomChoiceResize,
    RandomFlip,
    RandomGrayscale,
    RandomResize,
    Resize,
    TestTimeAug,
)
from mmcv.transforms.wrappers import Compose, KeyMapper, RandomApply, RandomChoice, TransformBroadcaster

try:
    import torch  # noqa: F401
except ImportError:
    __all__ = [
        'TRANSFORMS',
        'BaseTransform',
        'CenterCrop',
        'Compose',
        'KeyMapper',
        'LoadAnnotations',
        'LoadImageFromFile',
        'MultiScaleFlipAug',
        'Normalize',
        'Pad',
        'RandomApply',
        'RandomChoice',
        'RandomChoiceResize',
        'RandomFlip',
        'RandomGrayscale',
        'RandomResize',
        'Resize',
        'TestTimeAug',
        'TransformBroadcaster'
    ]
else:
    from mmcv.transforms.formatting import ImageToTensor, ToTensor, to_tensor

    __all__ = [
        'TRANSFORMS',
        'BaseTransform',
        'CenterCrop',
        'Compose',
        'ImageToTensor',
        'KeyMapper',
        'LoadAnnotations',
        'LoadImageFromFile',
        'MultiScaleFlipAug',
        'Normalize',
        'Pad',
        'RandomApply',
        'RandomChoice',
        'RandomChoiceResize',
        'RandomFlip',
        'RandomGrayscale',
        'RandomResize',
        'Resize',
        'TestTimeAug',
        'ToTensor',
        'TransformBroadcaster',
        'to_tensor'
    ]
