# Copyright (c) OpenMMLab. All rights reserved.
# This file forwards imports from mmcv.ops.roi_align to maintain compatibility
# with code that imports directly from this module.

from mmcv.ops.roi_align..roi_align import RoIAlign, roi_align

__all__ = ['RoIAlign', 'roi_align']