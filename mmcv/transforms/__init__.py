# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseTransform
from .builder import TRANSFORMS
from .loading import LoadAnnotations, LoadImageFromFile
from .processing import (CenterCrop, MultiScaleFlipAug, Normalize, Pad,
                         RandomChoiceResize, RandomFlip, RandomGrayscale,
                         RandomResize, Resize)
from .wrappers import Compose, KeyMapper, RandomChoice, TransformBroadcaster

try:
    import torch  # noqa: F401
except ImportError:
    __all__ = [
        'BaseTransform', 'TRANSFORMS', 'TransformBroadcaster', 'Compose',
        'RandomChoice', 'KeyMapper', 'LoadImageFromFile', 'LoadAnnotation',
        'Normalize', 'Resize', 'Pad', 'RandomFlip', 'RandomChoiceResize',
        'CenterCrop', 'RandomGrayscale', 'MultiScaleFlipAug', 'RandomResize'
    ]
else:
    from .formatting import ImageToTensor, ToTensor, to_tensor

    __all__ = [
        'BaseTransform', 'TRANSFORMS', 'TransformBroadcaster', 'Compose',
        'RandomChoice', 'KeyMapper', 'LoadImageFromFile', 'LoadAnnotation',
        'Normalize', 'Resize', 'Pad', 'ToTensor', 'to_tensor', 'ImageToTensor',
        'RandomFlip', 'RandomChoiceResize', 'CenterCrop', 'RandomGrayscale',
        'MultiScaleFlipAug', 'RandomResize'
    ]
