# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.utils import IS_MLU_AVAILABLE

from mmcv.ops.assign_score_withk import assign_score_withk
from mmcv.ops.bbox import bbox_overlaps
from mmcv.ops.bezier_align import BezierAlign, bezier_align
from mmcv.ops.bias_act import bias_act
from mmcv.ops.border_align import BorderAlign, border_align
from mmcv.ops.box_iou_quadri import box_iou_quadri
from mmcv.ops.carafe import CARAFE, CARAFENaive, CARAFEPack, carafe, carafe_naive
from mmcv.ops.cc_attention import CrissCrossAttention
from mmcv.ops.chamfer_distance import chamfer_distance
from mmcv.ops.contour_expand import contour_expand
from mmcv.ops.conv2d_gradfix import conv2d, conv_transpose2d
from mmcv.ops.convex_iou import convex_giou, convex_iou
from mmcv.ops.corner_pool import CornerPool
from mmcv.ops.correlation import Correlation
from mmcv.ops.deform_conv import DeformConv2d, DeformConv2dPack, deform_conv2d
from mmcv.ops.deform_roi_pool import DeformRoIPool, DeformRoIPoolPack, ModulatedDeformRoIPoolPack, deform_roi_pool
from mmcv.ops.deprecated_wrappers import Conv2d_deprecated as Conv2d
from mmcv.ops.deprecated_wrappers import ConvTranspose2d_deprecated as ConvTranspose2d
from mmcv.ops.deprecated_wrappers import Linear_deprecated as Linear
from mmcv.ops.deprecated_wrappers import MaxPool2d_deprecated as MaxPool2d
from mmcv.ops.filtered_lrelu import filtered_lrelu
from mmcv.ops.focal_loss import SigmoidFocalLoss, SoftmaxFocalLoss, sigmoid_focal_loss, softmax_focal_loss
from mmcv.ops.furthest_point_sample import furthest_point_sample, furthest_point_sample_with_dist
from mmcv.ops.fused_bias_leakyrelu import FusedBiasLeakyReLU, fused_bias_leakyrelu
from mmcv.ops.gather_points import gather_points
from mmcv.ops.group_points import GroupAll, QueryAndGroup, grouping_operation
from mmcv.ops.info import get_compiler_version, get_compiling_cuda_version
from mmcv.ops.knn import knn
from mmcv.ops.masked_conv import MaskedConv2d, masked_conv2d
from mmcv.ops.min_area_polygons import min_area_polygons
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2d, ModulatedDeformConv2dPack, modulated_deform_conv2d
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmcv.ops.nms import batched_nms, nms, nms_match, nms_quadri, nms_rotated, soft_nms
from mmcv.ops.pixel_group import pixel_group
from mmcv.ops.point_sample import SimpleRoIAlign, point_sample, rel_roi_point_to_rel_img_point
from mmcv.ops.points_in_polygons import points_in_polygons
from mmcv.ops.points_sampler import PointsSampler
from mmcv.ops.prroi_pool import PrRoIPool, prroi_pool
from mmcv.ops.psa_mask import PSAMask

# Import roi_align functionality from the roi_align module
from mmcv.ops.roi_align import RoIAlign, roi_align
from mmcv.ops.roi_pool import RoIPool, roi_pool
from mmcv.ops.roipoint_pool3d import RoIPointPool3d
from mmcv.ops.rotated_feature_align import rotated_feature_align
from mmcv.ops.saconv import SAConv2d
from mmcv.ops.sparse_conv import (
    SparseConv2d,
    SparseConv3d,
    SparseConvTranspose2d,
    SparseConvTranspose3d,
    SparseInverseConv2d,
    SparseInverseConv3d,
    SubMConv2d,
    SubMConv3d,
)
from mmcv.ops.sparse_modules import SparseModule, SparseSequential
from mmcv.ops.sparse_pool import SparseMaxPool2d, SparseMaxPool3d
from mmcv.ops.sparse_structure import SparseConvTensor, scatter_nd
from mmcv.ops.sync_bn import SyncBatchNorm
from mmcv.ops.three_interpolate import three_interpolate
from mmcv.ops.three_nn import three_nn
from mmcv.ops.tin_shift import TINShift, tin_shift
from mmcv.ops.upfirdn2d import filter2d, upfirdn2d, upsample2d

__all__ = [
    'CARAFE',
    'BezierAlign',
    'BorderAlign',
    'CARAFENaive',
    'CARAFEPack',
    'Conv2d',
    'ConvTranspose2d',
    'CornerPool',
    'Correlation',
    'CrissCrossAttention',
    'DeformConv2d',
    'DeformConv2dPack',
    'DeformRoIPool',
    'DeformRoIPoolPack',
    'FusedBiasLeakyReLU',
    'GroupAll',
    'Linear',
    'MaskedConv2d',
    'MaxPool2d',
    'ModulatedDeformConv2d',
    'ModulatedDeformConv2dPack',
    'ModulatedDeformRoIPoolPack',
    'MultiScaleDeformableAttention',
    'PSAMask',
    'PointsSampler',
    'PrRoIPool',
    'QueryAndGroup',
    'RoIAlign',
    'RoIPointPool3d',
    'RoIPool',
    'SAConv2d',
    'SigmoidFocalLoss',
    'SimpleRoIAlign',
    'SoftmaxFocalLoss',
    'SparseConv2d',
    'SparseConv3d',
    'SparseConvTensor',
    'SparseConvTranspose2d',
    'SparseConvTranspose3d',
    'SparseInverseConv2d',
    'SparseInverseConv3d',
    'SparseMaxPool2d',
    'SparseMaxPool3d',
    'SparseModule',
    'SparseSequential',
    'SubMConv2d',
    'SubMConv3d',
    'SyncBatchNorm',
    'TINShift',
    'assign_score_withk',
    'batched_nms',
    'bbox_overlaps',
    'bezier_align',
    'bias_act',
    'border_align',
    'box_iou_quadri',
    'carafe',
    'carafe_naive',
    'chamfer_distance',
    'contour_expand',
    'conv2d',
    'conv_transpose2d',
    'convex_giou',
    'convex_iou',
    'deform_conv2d',
    'deform_roi_pool',
    'filter2d',
    'filtered_lrelu',
    'furthest_point_sample',
    'furthest_point_sample_with_dist',
    'fused_bias_leakyrelu',
    'gather_points',
    'get_compiler_version',
    'get_compiling_cuda_version',
    'grouping_operation',
    'knn',
    'masked_conv2d',
    'min_area_polygons',
    'modulated_deform_conv2d',
    'nms',
    'nms_match',
    'nms_quadri',
    'nms_rotated',
    'pixel_group',
    'point_sample',
    'points_in_polygons',
    'prroi_pool',
    'rel_roi_point_to_rel_img_point',
    'roi_align',
    'roi_pool',
    'rotated_feature_align',
    'scatter_nd',
    'sigmoid_focal_loss',
    'soft_nms',
    'softmax_focal_loss',
    'three_interpolate',
    'three_nn',
    'tin_shift',
    'upfirdn2d',
    'upsample2d'
]

if IS_MLU_AVAILABLE:
    from mmcv.ops.deform_conv import DeformConv2dPack_MLU  # noqa:F401
    from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack_MLU  # noqa:F401
    __all__.extend(['DeformConv2dPack_MLU', 'ModulatedDeformConv2dPack_MLU'])
