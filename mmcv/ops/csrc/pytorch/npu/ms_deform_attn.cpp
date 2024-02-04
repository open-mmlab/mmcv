
#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

Tensor ms_deform_attn_impl_backward_npu(const Tensor &value,
                                   const Tensor &spatial_shapes,
                                   const Tensor &level_start_index,
                                   const Tensor &sampling_loc,
                                   const Tensor &attn_weight,
                                   const int im2col_step) {
    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnFunction,
                 value, spatial_shapes, level_start_index, sampling_loc, attn_weight);
}

void ms_deform_attn_impl_backward_npu(
    const Tensor &value, const Tensor &spatial_shapes,
    const Tensor &level_start_index, const Tensor &sampling_loc,
    const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
    Tensor &grad_sampling_loc, Tensor &grad_attn_weight,
    const int im2col_step) {
    EXEC_NPU_CMD(aclnnMultiScaleDeformableAttnFunctionGrad,
                 value, spatial_shapes, level_start_index, sampling_loc, attn_weight,
                 grad_output, grad_value, grad_sampling_loc, grad_attn_weight);
}

Tensor ms_deform_attn_impl_backward_npu(const Tensor &value,
                                        const Tensor &spatial_shapes,
                                        const Tensor &level_start_index,
                                        const Tensor &sampling_loc,
                                        const Tensor &attn_weight,
                                        const int im2col_step);
REGISTER_NPU_IMPL(ms_deform_attn_impl_backward, ms_deform_attn_impl_backward_npu);

void ms_deform_attn_impl_backward(const Tensor &value, const Tensor &spatial_shapes,
                                  const Tensor &level_start_index, const Tensor &sampling_loc,
                                  const Tensor &attn_weight, const Tensor &grad_output, Tensor &grad_value,
                                  Tensor &grad_sampling_loc, Tensor &grad_attn_weight,
                                  const int im2col_step);
REGISTER_NPU_IMPL(ms_deform_attn_impl_backward,
                  ms_deform_attn_impl_backward_npu);
