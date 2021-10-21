// Copyright (c) 2021, SenseTime.

#include "parrots_camb_utils.h"

#ifdef PARROTS_USE_CAMB

using namespace parrots;

// policy function
static void policyFunc(cnrtDim3_t* k_dim, cnrtFunctionType_t* k_type, const DArrayLite& input,
                       const DArrayLite& target, const DArrayLite& weight) {
    auto N = input.dim(0);
    auto C = input.dim(1);

    auto nram_size = getDeviceAttr(cnrtAttrNramSizePerMcore);
    auto c_align_size = PAD_UP((C * itemsize(input)), NFU_ALIGN_SIZE);
    const int split_target_num = 2;
    const int split_pipeline_num = 6;
    auto scalar_size = NFU_ALIGN_SIZE;
    auto weight_size = c_align_size;
    const int target_data_width =
        target.elemType() == Prim::Int64 ? itemsize(target) / 2 : itemsize(target);

    // n_seg * c_align_size * split_pipeline_num + n_seg * target.itemsize() *
    // split_target_num
    //     + weight_size + scalar_size <= nram_size
    auto n_seg = (nram_size - weight_size - scalar_size) /
                 (c_align_size * split_pipeline_num + target_data_width * split_target_num);
    auto seg_num = (N + n_seg - 1) / n_seg;

    auto core_dim = getDeviceAttr(cnrtAttrMcorePerCluster);
    auto cluster_num = getDeviceAttr(cnrtAttrClusterCount);
    auto core_num = core_dim * cluster_num;

    k_dim->x = *k_type;
    k_dim->y = seg_num > core_num ? cluster_num : (seg_num + core_dim - 1) / core_dim;
    k_dim->z = 1;
}

void KernelFocalLossSigmoidForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type, cnrtQueue_t queue,
                                   cnrtDataType_t d_type, const void* input, const void* target,
                                   const void* weight, const int32_t N, const int32_t C,
                                   const float alpha, const float gamma, void* output);

void sigmoidFocalLossForwardMLUKernelLauncher(CambContext& ctx, const DArrayLite& input,
                                              const DArrayLite& target, const DArrayLite& weight,
                                              DArrayLite& output, float gamma, float alpha) {
    // params check
    PARROTS_CHECKARGS(gamma >= 0) << "gamma should be greater than or equal to 0. "
                                  << "But now gamma is " << gamma << ".";

    // check dtype
    PARROTS_CHECKARGS((input.elemType() == Prim::Float32) || (input.elemType() == Prim::Float16))
        << "Data type of input should be Float or Half. But now input type is " << input.elemType()
        << ".";

    PARROTS_CHECKARGS(target.elemType() == Prim::Int32)
        << "target type should be int 32. But now target type is " << target.elemType() << ".";

    PARROTS_CHECKARGS(output.elemType() == input.elemType())
        << "Data types of input and output should be the same. But now input "
           "type is "
        << input.elemType() << ", output type is " << output.elemType() << ".";

    // check weight
    if (weight.size() > 0) {
        PARROTS_CHECKARGS(weight.elemType() == input.elemType())
            << "Data types of input and weight should be the same. But now "
               "input type is "
            << input.elemType() << ", weight type is " << weight.elemType() << ".";
    }
    // check C
    auto nram_size = getDeviceAttr(cnrtAttrNramSizePerMcore);
    auto input_N = input.dim(0);
    auto input_C = input.dim(1);
    auto split_target_num = 2;
    int split_pipeline_num = 6;
    const int has_weight = (int)(weight.size() > 0);
    const int target_data_width =
        target.elemType() == Prim::Int64 ? itemsize(target) / 2 : itemsize(target);
    // target supports only INT on MLU device
    // while it keeps LONG on host side, so target.itemsize()/2
    auto threshold_C =
        PAD_DOWN((nram_size - NFU_ALIGN_SIZE - split_target_num * target_data_width) /
                     (split_pipeline_num + has_weight),
                 NFU_ALIGN_SIZE) /
        itemsize(input);
    PARROTS_CHECKARGS(threshold_C >= input_C)
        << "input.size(1) should be in the range of [0, " << threshold_C
        << "], but now input.dim(1) is " << input_C << ".";

    if (input.size() == 0 || target.size() == 0 || output.size() == 0) {
        // return if zero-element
        return;
    }

    // calculate task dimension
    cnrtDim3_t k_dim;
    cnrtFunctionType_t k_type = CNRT_FUNC_TYPE_UNION1;
    policyFunc(&k_dim, &k_type, input, target, weight);
    auto core_dim = getDeviceAttr(cnrtAttrMcorePerCluster);

    // get compute queue
    auto queue = getStreamNative<CambDevice>(ctx.getStream());
    auto input_ptr = input.data();
    auto target_ptr = target.data();
    auto weight_ptr = weight.size() > 0 ? weight.data() : nullptr;
    auto output_ptr = output.data();
    // get dtype of input
    cnrtDataType_t d_type = getCnrtDataType(input.elemType());

    // launch kernel
    KernelFocalLossSigmoidForward(k_dim, k_type, queue, d_type, input_ptr, target_ptr, weight_ptr,
                                  input_N, input_C, alpha, gamma, output_ptr);
}

void sigmoid_focal_loss_forward_camb_parrots(CambContext& ctx,
                                             const SSElement& attr,
                                             const OperatorBase::in_list_t& ins,
                                             OperatorBase::out_list_t& outs) {
    float gamma;
    float alpha;
    SSAttrs(attr).get<float>("gamma", gamma).get<float>("alpha", alpha).done();

    // get inputs and outputs
    const auto& input = ins[0];
    const auto& target = ins[1];
    const auto& weight = ins[2];
    auto& output = outs[0];

    sigmoidFocalLossForwardMLUKernelLauncher(ctx, input, target, weight, output,
                                             gamma, alpha);
}

PARROTS_EXTENSION_REGISTER(sigmoid_focal_loss_forward)
    .attr("gamma")
    .attr("alpha")
    .input(3)
    .output(1)
    #ifdef PARROTS_USE_CAMB
    .apply(sigmoid_focal_loss_forward_camb_parrots)
    #endif // PARROTS_USE_CAMB
    .done();

#endif  // PARROTS_USE_CAMB
