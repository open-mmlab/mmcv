# Copyright (c) OpenMMLab. All rights reserved.
from .checkpoint import CheckpointHook
from .closure import ClosureHook
from .ema import EMAHook
from .evaluation import DistEvalHook, EvalHook
from .hook import HOOKS, Hook
from .iter_timer import IterTimerHook
from .logger import (DvcliveLoggerHook, LoggerHook, MlflowLoggerHook,
                     NeptuneLoggerHook, PaviLoggerHook, TensorboardLoggerHook,
                     TextLoggerHook, WandbLoggerHook)
from .lr_updater import (CosineAnnealingLrUpdaterHook,
                         CosineRestartLrUpdaterHook, CyclicLrUpdaterHook,
                         ExpLrUpdaterHook, FixedLrUpdaterHook,
                         FlatCosineAnnealingLrUpdaterHook, InvLrUpdaterHook,
                         LrUpdaterHook, OneCycleLrUpdaterHook,
                         PolyLrUpdaterHook, StepLrUpdaterHook)
from .memory import EmptyCacheHook
from .momentum_updater import (CosineAnnealingMomentumUpdaterHook,
                               CyclicMomentumUpdaterHook, MomentumUpdaterHook,
                               OneCycleMomentumUpdaterHook,
                               StepMomentumUpdaterHook)
from .optimizer import (Fp16OptimizerHook, GradientCumulativeFp16OptimizerHook,
                        GradientCumulativeOptimizerHook, OptimizerHook)
from .profiler import ProfilerHook
from .sampler_seed import DistSamplerSeedHook
from .sync_buffer import SyncBuffersHook

__all__ = [
    'HOOKS', 'Hook', 'CheckpointHook', 'ClosureHook', 'LrUpdaterHook',
    'FixedLrUpdaterHook', 'StepLrUpdaterHook', 'ExpLrUpdaterHook',
    'PolyLrUpdaterHook', 'InvLrUpdaterHook', 'CosineAnnealingLrUpdaterHook',
    'FlatCosineAnnealingLrUpdaterHook', 'CosineRestartLrUpdaterHook',
    'CyclicLrUpdaterHook', 'OneCycleLrUpdaterHook', 'OptimizerHook',
    'Fp16OptimizerHook', 'IterTimerHook', 'DistSamplerSeedHook',
    'EmptyCacheHook', 'LoggerHook', 'MlflowLoggerHook', 'PaviLoggerHook',
    'TextLoggerHook', 'TensorboardLoggerHook', 'NeptuneLoggerHook',
    'WandbLoggerHook', 'DvcliveLoggerHook', 'MomentumUpdaterHook',
    'StepMomentumUpdaterHook', 'CosineAnnealingMomentumUpdaterHook',
    'CyclicMomentumUpdaterHook', 'OneCycleMomentumUpdaterHook',
    'SyncBuffersHook', 'EMAHook', 'EvalHook', 'DistEvalHook', 'ProfilerHook',
    'GradientCumulativeOptimizerHook', 'GradientCumulativeFp16OptimizerHook'
]
