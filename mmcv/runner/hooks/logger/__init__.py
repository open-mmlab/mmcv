# Copyright (c) Open-MMLab. All rights reserved.
from .base import LoggerHook
from .pavi import PaviLoggerHook
from .tensorboard import TensorboardLoggerHook
from .text import TextLoggerHook
from .wandb import WandbLoggerHook

__all__ = [
    'LoggerHook', 'PaviLoggerHook', 'TensorboardLoggerHook', 'TextLoggerHook',
    'WandbLoggerHook'
]
