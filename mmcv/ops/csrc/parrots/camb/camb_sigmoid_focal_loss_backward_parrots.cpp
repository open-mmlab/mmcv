// Copyright (c) 2021, SenseTime.

#include "parrots_camb_utils.h"

#ifdef PARROTS_USE_CAMB

using namespace parrots;

// Policy Function
static void policyFunc(cnrtDim3_t* k_dim, cnrtFunctionType_t* k_type) {
  // set Union1 Job
  *k_type = CNRT_FUNC_TYPE_UNION1;
  k_dim->x = getDeviceAttr(cnrtAttrMcorePerCluster);
  k_dim->y = getDeviceAttr(cnrtAttrClusterCount);
  k_dim->z = 1;
}

void getDealNAndThresholdC(const int compute_data_bytes,
                           const int target_data_bytes, const int total_c,
                           int* deal_n_ptr, int* threshold_c_ptr,
                           const bool has_weight, const bool is_half) {
  /* NRAM partition:
   *
   * |-----------------ping pong---------------------|
   * | input | pt | alpha_t | temp | output | target | flt_min | gamma |
   * weight |
   *
   * split_pipeline_num is 5: including input, pt, alpha_t, temp, output.
   */
  const int nram_split_num = 5;
  const int nram_split_pingpong = 2;
  // const int max_nram_size = getDeviceAttr(cnrtAttrNramSizePerMcore);
  const int max_nram_size = MAX_NRAM_SIZE;

  int32_t compute_align_size = NFU_ALIGN_SIZE;
  if (is_half) {
    compute_align_size += NFU_ALIGN_SIZE;
  }
  const int32_t compute_align_num = compute_align_size / compute_data_bytes;
  // reservered_align_size: including input(ping pong), pt(ping pong),
  //                        alpha_t(ping pong), temp(ping pong), output(ping
  //                        pong), target(ping pong), flt_min and gamma.
  const int reservered_align_size =
      ((nram_split_num + 1) * nram_split_pingpong + 2) * compute_align_size;
  int nram_pingpong_size = max_nram_size - reservered_align_size;

  int compute_c = total_c;
  int threshold_c = 0;
  if (has_weight) {
    // reserved space for weight to align
    nram_pingpong_size -= NFU_ALIGN_SIZE;

    // threshold_c * nram_split_pingpong * compute_data_bytes *
    // nram_split_num + nram_split_pingpong * target_data_bytes +
    // threshold_c * compute_data_bytes <= nram_pingpong_size
    threshold_c =
        (nram_pingpong_size - nram_split_pingpong * target_data_bytes) /
        (compute_data_bytes * (nram_split_num * nram_split_pingpong + 1));
    threshold_c = PAD_DOWN(threshold_c, compute_align_num);
    int weight_space = PAD_UP(total_c * compute_data_bytes, NFU_ALIGN_SIZE);

    // reserved space for weight
    nram_pingpong_size -= weight_space;
    compute_c = PAD_UP(total_c, compute_align_num);
  } else {
    // threshold_c * nram_split_pingpong * compute_data_bytes *
    // nram_split_num + nram_split_pingpong * target_data_bytes <=
    // nram_pingpong_size
    threshold_c =
        (nram_pingpong_size / nram_split_pingpong - target_data_bytes) /
        (nram_split_num * compute_data_bytes);
  }
  // deal_n * compute_c * nram_split_pingpong * compute_data_bytes *
  // nram_split_num + deal_n * nram_split_pingpong * target_data_bytes <=
  // nram_pingpong_size
  *deal_n_ptr =
      nram_pingpong_size /
      ((nram_split_num * compute_c * compute_data_bytes + target_data_bytes) *
       nram_split_pingpong);
  *threshold_c_ptr = threshold_c;
}

void KernelFocalLossSigmoidBackward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
                                    cnrtQueue_t queue,
                                    const cnrtDataType_t d_type,
                                    const void* input, const void* target,
                                    const void* weight, const float gamma,
                                    const float alpha, const int32_t dim_n,
                                    const int32_t deal_n, const int32_t dim_c,
                                    void* output);

void SigmoidFocalLossBackwardMLUKernelLauncher(
    CambContext& ctx, const DArrayLite& input, const DArrayLite& target,
    const DArrayLite& weight, DArrayLite& output, const float gamma,
    const float alpha) {
  // params check
  PARROTS_CHECKARGS(gamma >= 0)
      << "gamma should be greater than or equal to 0. "
      << "But now gamma is " << gamma << ".";

  // check dtype
  PARROTS_CHECKARGS((input.elemType() == Prim::Float32) ||
                    (input.elemType() == Prim::Float16))
      << "Data type of input should be Float or Half. But now input type is "
      << input.elemType() << ".";

  PARROTS_CHECKARGS(target.elemType() == Prim::Int32)
      << "target type should be int 32. But now target type is "
      << target.elemType() << ".";

  PARROTS_CHECKARGS(output.elemType() == input.elemType())
      << "Data types of input and output should be the same. But now input "
         "type is "
      << input.elemType() << ", output type is " << output.elemType() << ".";

  bool has_weight = false;
  // check weight
  if (weight.size() > 0) {
    PARROTS_CHECKARGS(weight.elemType() == input.elemType())
        << "Data types of input and weight should be the same. But now "
           "input type is "
        << input.elemType() << ", weight type is " << weight.elemType() << ".";
    has_weight = true;
  }

  auto dim_c = input.dim(1);
  const int compute_data_bytes = sizeof(float);
  // target only supports INT on MLU device,
  // while it keeps LONG on host side, so target.itemsize() / 2.
  const int target_data_bytes = itemsize(target);

  int deal_n = 0;
  int threshold_c = 0;
  bool is_half = false;
  if (input.elemType() == Prim::Float16) {
    is_half = true;
  }
  // calculate deal_n and threshold_c
  getDealNAndThresholdC(compute_data_bytes, target_data_bytes, dim_c, &deal_n,
                        &threshold_c, has_weight, is_half);

  // check C
  PARROTS_CHECKARGS(threshold_c >= dim_c)
      << "input.dim(1) should be in the range of [0, " << threshold_c
      << "], but now input.dim(1) is " << dim_c << ".";

  if (input.size() == 0 || target.size() == 0 || output.size() == 0) {
    // return if zero-element
    return;
  }

  // set task dimension
  cnrtDim3_t k_dim;
  cnrtFunctionType_t k_type;
  policyFunc(&k_dim, &k_type);

  // get compute queue
  auto queue = getStreamNative<CambDevice>(ctx.getStream());

  // get ptr of tensors
  auto input_ptr = input.data();
  auto target_ptr = target.data();
  auto weight_ptr = has_weight ? weight.data() : nullptr;
  auto* output_ptr = output.data();

  // get dtype of input
  cnrtDataType_t d_type = getCnrtDataType(input.elemType());
  auto core_dim = getDeviceAttr(cnrtAttrMcorePerCluster);
  auto dim_n = input.dim(0);

  // launch kernel
  KernelFocalLossSigmoidBackward(k_dim, k_type, queue, d_type, input_ptr,
                                 target_ptr, weight_ptr, gamma, alpha, dim_n,
                                 deal_n, dim_c, output_ptr);
}

void sigmoid_focal_loss_backward_camb_parrots(
    CambContext& ctx, const SSElement& attr, const OperatorBase::in_list_t& ins,
    OperatorBase::out_list_t& outs) {
  float gamma;
  float alpha;
  SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

  // get inputs and outputs
  const auto& input = ins[0];
  const auto& target = ins[1];
  const auto& weight = ins[2];

  auto& grad_input = outs[0];

  SigmoidFocalLossBackwardMLUKernelLauncher(ctx, input, target, weight,
                                            grad_input, gamma, alpha);
}

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_backward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    .apply(sigmoid_focal_loss_backward_camb_parrots)
    .done();

#endif  // PARROTS_USE_CAMB
