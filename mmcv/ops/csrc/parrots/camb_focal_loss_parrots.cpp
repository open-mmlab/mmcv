#include <parrots/extension.hpp>
#ifdef PARROTS_USE_CAMB

using namespace parrots;
#include "./mlu_utils.h"
#include <cnrt.h>
cnrtDataType_t getCnrtDataType(parrots::ValueType vt) {
    switch (vt.prim()) {
        case Prim::Float16:
            return cnrtFloat16;
        case Prim::Float32:
            return cnrtFloat32;
        case Prim::Int16:
            return cnrtInt16;
        case Prim::Int32:
            return cnrtInt32;
        case Prim::Int8:
            return cnrtInt8;
        case Prim::Uint8:
            return cnrtUInt8;
        case Prim::Bool:
            return cnrtBool;
        default:
            PARROTS_NOTSUPPORTED << "Unsupported data type for CNRT: "
                                 << vt.name();
    }
}

int getDeviceAttr(const cnrtDeviceAttr_t& attr) {
    int ordinal = 0;
    cnrtGetDevice(&ordinal);
    int value = 0;
    cnrtDeviceGetAttribute(&value, attr, ordinal);
    return value;
}

// policy function
static void policyFunc(cnrtDim3_t* k_dim, cnrtFunctionType_t* k_type,
        const DArrayLite& input, const DArrayLite& target,
        const DArrayLite& weight) {
    auto N = input.dim(0);
    auto C = input.dim(1);
    auto itemsize = [](const DArrayLite& input) -> int {
        int itemSize = 0;
        if (input.size() > 0) {
            itemSize = input.nbytes() / input.size();
        }
        return itemSize;
    };

    auto nram_size = getDeviceAttr(cnrtAttrNramSizePerMcore);
    auto c_align_size = PAD_UP((C * itemsize(input)), NFU_ALIGN_SIZE);
    const int split_target_num = 2;
    const int split_pipeline_num = 6;
    auto scalar_size = NFU_ALIGN_SIZE;
    auto weight_size = c_align_size;
    const int target_data_width = target.elemType() == Prim::Int64
                                      ? itemsize(target) / 2
                                      : itemsize(target);

    // n_seg * c_align_size * split_pipeline_num + n_seg * target.itemsize() *
    // split_target_num
    //     + weight_size + scalar_size <= nram_size
    auto n_seg = (nram_size - weight_size - scalar_size) /
                 (c_align_size * split_pipeline_num +
                     target_data_width * split_target_num);
    auto seg_num = (N + n_seg - 1) / n_seg;

    auto core_dim = getDeviceAttr(cnrtAttrMcorePerCluster);
    auto cluster_num = getDeviceAttr(cnrtAttrClusterCount);
    auto core_num = core_dim * cluster_num;

    k_dim->x = *k_type;
    k_dim->y =
        seg_num > core_num ? cluster_num : (seg_num + core_dim - 1) / core_dim;
    k_dim->z = 1;
}

void checkFocalLossSigmoidForwardValidation(const DArrayLite& input,
        const DArrayLite& target, const DArrayLite& weight,
        const DArrayLite& output) {
    // check shape
    PARROTS_CHECKARGS(input.ndims() == 2)
        << "Dimension num of input should be 2.But now is " << input.ndims()
        << ".";

    PARROTS_CHECKARGS(output.ndims() == 2)
        << "Dimension num of output should be 2.But now is " << output.ndims()
        << ".";

    PARROTS_CHECKARGS(target.ndims() == 1)
        << "Dimension num of target should be 1. But now is " << target.ndims()
        << ".";

    PARROTS_CHECKARGS(input.dim(0) == target.dim(0))
        << "Element num of target should be " << input.dim(0) << ". But now is "
        << target.dim(0) << ".";

    PARROTS_CHECKARGS(
        input.dim(0) == output.dim(0) && input.dim(1) == output.dim(1))
        << "Shape of output and input must be euqal, but now output is "
        << output.dim(0) << ", " << output.dim(1) << " and input is "
        << input.dim(0) << ", " << input.dim(1) << ".";

    // check dtype
    PARROTS_CHECKARGS(
        input.elemType() == Prim::Float32 || input.elemType() == Prim::Float16)
        << "Data type of input should be Float or Half. But now input type is "
        << input.elemType() << ".";

    PARROTS_CHECKARGS(target.elemType() == Prim::Int64)
        << "target type should be Long. But now target type is "
        << target.elemType() << ".";

    PARROTS_CHECKARGS(output.elemType() == input.elemType())
        << "Data types of input and output should be the same. But now input "
           "type is "
        << input.elemType() << ", output type is " << output.elemType() << ".";

    // check weight
    if (weight.data() != nullptr) {
        PARROTS_CHECKARGS(weight.elemType() == input.elemType())
            << "Data types of input and weight should be the same. But now "
               "input type is "
            << input.elemType() << ", weight type is " << weight.elemType()
            << ".";

        PARROTS_CHECKARGS(weight.ndims() == 1)
            << "Dimension num of weight should be 1. But now is "
            << weight.ndims() << ".";

        PARROTS_CHECKARGS(weight.dim(0) == input.dim(1))
            << "Element num of weight should be " << input.dim(1)
            << ". But now is " << weight.dim(0) << ".";
    }
}

void KernelFocalLossSigmoidForward(cnrtDim3_t k_dim, cnrtFunctionType_t k_type,
        cnrtQueue_t queue, cnrtDataType_t d_type, const void* input,
        const void* target, const void* weight, const int32_t N,
        const int32_t C, const float alpha, const float gamma, void* output);

void sigmoidFocalLossForwardMLUKernelLauncher(CambContext& ctx,
        const DArrayLite& input, const DArrayLite& target,
        const DArrayLite& weight, DArrayLite& output, float gamma,
        float alpha) {
    // params check
    PARROTS_CHECKARGS(gamma >= 0)
        << "gamma should be greater than or equal to 0. "
        << "But now gamma is " << gamma << ".";

    checkFocalLossSigmoidForwardValidation(input, target, weight, output);
    // check C
    auto input_N = input.dim(0);
    auto input_C = input.dim(1);
    auto split_target_num = 2;
    int split_pipeline_num = 6;
    auto nram_size = getDeviceAttr(cnrtAttrNramSizePerMcore);

    // target supports only INT on MLU device
    // while it keeps LONG on host side, so target.itemsize()/2
    auto threshold_C =
        PAD_DOWN((nram_size - NFU_ALIGN_SIZE -
                     split_target_num * (target.nbytes() / target.size() / 2)) /
                     split_pipeline_num,
            NFU_ALIGN_SIZE) /
        (input.nbytes() / input.size());

    PARROTS_CHECKARGS(threshold_C >= input_C)
        << "input.size(1) should be in the range of [0, " << threshold_C
        << "]. ",
        "But now input.size(1) is ", input_C, ".";

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
    auto weight_ptr = weight.data();
    weight_ptr = nullptr;
    auto output_ptr = output.data();
    // get dtype of input
    cnrtDataType_t d_type = getCnrtDataType(input.elemType());

    // launch kernel
    KernelFocalLossSigmoidForward(k_dim, k_type, queue, d_type, input_ptr,
        target_ptr, weight_ptr, input_N, input_C, alpha, gamma, output_ptr);
    cnrtSyncQueue(queue);
}

#endif  // PARROTS_USE_CAMB
