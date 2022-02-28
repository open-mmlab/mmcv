# Copyright (c) OpenMMLab. All rights reserved.
from .builder import TRANSFORMS
from .loading import LoadAnnotation, LoadImageFromFile
from .processing import (CenterCrop, MultiScaleFlipAug, Normalize, Pad,
                         RandomFlip, RandomGrayscale, RandomMultiscaleResize,
                         RandomResize, Resize)
from .wrappers import ApplyToMultiple, Compose, RandomChoice, Remap

try:
    import torch  # noqa: F401
except ImportError:
    __all__ = [
        'TRANSFORMS', 'ApplyToMultiple', 'Compose', 'RandomChoice', 'Remap',
        'LoadImageFromFile', 'LoadAnnotation', 'Normalize', 'Resize', 'Pad',
        'RandomFlip', 'RandomMultiscaleResize', 'CenterCrop',
        'RandomGrayscale', 'MultiScaleFlipAug', 'RandomResize'
    ]
else:
    from .formatting import ImageToTensor, ToTensor, to_tensor

    __all__ = [
        'TRANSFORMS', 'ApplyToMultiple', 'Compose', 'RandomChoice', 'Remap',
        'LoadImageFromFile', 'LoadAnnotation', 'Normalize', 'Resize', 'Pad',
        'ToTensor', 'to_tensor', 'ImageToTensor', 'RandomFlip',
        'RandomMultiscaleResize', 'CenterCrop', 'RandomGrayscale',
        'MultiScaleFlipAug', 'RandomResize'
    ]
