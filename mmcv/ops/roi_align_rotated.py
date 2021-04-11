import torch.nn as nn
from torch.autograd import Function

from ..utils import ext_loader

ext_module = ext_loader.load_ext(
    '_ext', ['roi_align_rotated_forward', 'roi_align_rotated_backward'])


class RoIAlignRotatedFunction(Function):

    @staticmethod
    def symbolic(g, features, rois, out_size, spatial_scale, sample_num,
                 aligned, clockwise):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        return g.op(
            'mmcv::MMCVRoIAlignRotated',
            features,
            rois,
            output_height_i=out_h,
            output_width_i=out_h,
            spatial_scale_f=spatial_scale,
            sampling_ratio_i=sample_num,
            aligned_i=aligned,
            clockwise_i=clockwise)

    @staticmethod
    def forward(ctx,
                features,
                rois,
                out_size,
                spatial_scale,
                sample_num=0,
                aligned=True,
                clockwise=False):
        if isinstance(out_size, int):
            out_h = out_size
            out_w = out_size
        elif isinstance(out_size, tuple):
            assert len(out_size) == 2
            assert isinstance(out_size[0], int)
            assert isinstance(out_size[1], int)
            out_h, out_w = out_size
        else:
            raise TypeError(
                '"out_size" must be an integer or tuple of integers')
        ctx.spatial_scale = spatial_scale
        ctx.sample_num = sample_num
        ctx.aligned = aligned
        ctx.clockwise = clockwise
        ctx.save_for_backward(rois)
        ctx.feature_size = features.size()

        batch_size, num_channels, data_height, data_width = features.size()
        num_rois = rois.size(0)

        output = features.new_zeros(num_rois, num_channels, out_h, out_w)
        ext_module.roi_align_rotated_forward(
            features,
            rois,
            output,
            pooled_height=out_h,
            pooled_width=out_w,
            spatial_scale=spatial_scale,
            sample_num=sample_num,
            aligned=aligned,
            clockwise=clockwise)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        feature_size = ctx.feature_size
        spatial_scale = ctx.spatial_scale
        aligned = ctx.aligned
        clockwise = ctx.clockwise
        sample_num = ctx.sample_num
        rois = ctx.saved_tensors[0]
        assert feature_size is not None
        batch_size, num_channels, data_height, data_width = feature_size

        out_w = grad_output.size(3)
        out_h = grad_output.size(2)

        grad_input = grad_rois = None

        if ctx.needs_input_grad[0]:
            grad_input = rois.new_zeros(batch_size, num_channels, data_height,
                                        data_width)
            ext_module.roi_align_rotated_backward(
                grad_output.contiguous(),
                rois,
                grad_input,
                pooled_height=out_h,
                pooled_width=out_w,
                spatial_scale=spatial_scale,
                sample_num=sample_num,
                aligned=aligned,
                clockwise=clockwise)
        return grad_input, grad_rois, None, None, None, None, None


roi_align_rotated = RoIAlignRotatedFunction.apply


class RoIAlignRotated(nn.Module):

    def __init__(self,
                 out_size,
                 spatial_scale,
                 sample_num=0,
                 aligned=True,
                 clockwise=False):
        super(RoIAlignRotated, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)
        self.sample_num = int(sample_num)
        self.aligned = aligned
        self.clockwise = clockwise

    def forward(self, features, rois):
        return RoIAlignRotatedFunction.apply(features, rois, self.out_size,
                                             self.spatial_scale,
                                             self.sample_num, self.aligned,
                                             self.clockwise)
