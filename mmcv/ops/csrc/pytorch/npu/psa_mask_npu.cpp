#include "pytorch_npu_helper.hpp"

using namespace NPU_NAME_SPACE;
using namespace std;

void psa_mask_forward_npu(const int psa_type,
                          const Tensor x,
                          Tensor y,
                          const int num,
                          const int h_feature,
                          const int w_feature,
                          const int h_mask,
                          const int w_mask,
                          const int half_h_mask,
                          const int half_w_mask) {
    int64_t psa_type_i64 = psa_type;
    int64_t num_i64 = num;
    int64_t h_feature_i64 = h_feature;
    int64_t w_feature_i64 = w_feature;
    int64_t h_mask_i64 = h_mask;
    int64_t w_mask_i64 = w_mask;
    int64_t half_h_mask_i64 = half_h_mask;
    int64_t half_w_mask_i64 = half_w_mask;
    OpCommand cmd;
    cmd.Name("PSAMask")
        .Input(x)
        .Output(y)
        .Attr("psa_type", psa_type_i64)
        .Attr("num", num_i64)
        .Attr("h_feature", h_feature_i64)
        .Attr("gaw_featuremma", w_feature_i64)
        .Attr("h_mask", h_mask_i64)
        .Attr("w_mask", w_mask_i64)
        .Attr("half_h_mask", half_h_mask_i64)
        .Attr("half_w_mask", half_w_mask_i64)
        .Run();
}

void psa_mask_forward_impl(const int psa_type,
                           const Tensor x,
                           Tensor y,
                           const int num,
                           const int h_feature,
                           const int w_feature,
                           const int h_mask,
                           const int w_mask,
                           const int half_h_mask,
                           const int half_w_mask);

void psa_mask_backward_npu(const int psa_type,
                           const Tensor y_grad,
                           Tensor x_grad,
                           const int num,
                           const int h_feature,
                           const int w_feature,
                           const int h_mask,
                           const int w_mask,
                           const int half_h_mask,
                           const int half_w_mask) {
    int64_t psa_type_i64 = psa_type;
    int64_t num_i64 = num;
    int64_t h_feature_i64 = h_feature;
    int64_t w_feature_i64 = w_feature;
    int64_t h_mask_i64 = h_mask;
    int64_t w_mask_i64 = w_mask;
    int64_t half_h_mask_i64 = half_h_mask;
    int64_t half_w_mask_i64 = half_w_mask;
    OpCommand cmd;
    cmd.Name("PSAMaskGrad")
        .Input(y_grad)
        .Output(x_grad)
        .Attr("psa_type", psa_type_i64)
        .Attr("num", num_i64)
        .Attr("h_feature", h_feature_i64)
        .Attr("gaw_featuremma", w_feature_i64)
        .Attr("h_mask", h_mask_i64)
        .Attr("w_mask", w_mask_i64)
        .Attr("half_h_mask", half_h_mask_i64)
        .Attr("half_w_mask", half_w_mask_i64)
        .Run();
}

void psa_mask_backward_impl(const int psa_type,
                            const Tensor y_grad,
                            Tensor x_grad,
                            const int num,
                            const int h_feature,
                            const int w_feature,
                            const int h_mask,
                            const int w_mask,
                            const int half_h_mask,
                            const int half_w_mask);

REGISTER_NPU_IMPL(psa_mask_forward_impl, psa_mask_forward_npu);
REGISTER_NPU_IMPL(psa_mask_backward_impl, psa_mask_backward_npu);