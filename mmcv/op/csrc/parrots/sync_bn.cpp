#include <parrots/extension.hpp>
#include <parrots/foundation/ssattrs.hpp>
#include <parrots/foundation/exceptions.hpp>

using namespace parrots;

void sync_bn_forward_step1_cpu(HostContext& ctx,
                               const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
    PARROTS_NOT_IMPL << "Not support cpu sync bn forward step 1!";
}

void sync_bn_forward_step2_cpu(HostContext& ctx,
                               const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
    PARROTS_NOT_IMPL << "Not support cpu sync bn forward step 2!";
}

void sync_bn_forward_step3_cpu(HostContext& ctx,
                               const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
    PARROTS_NOT_IMPL << "Not support cpu sync bn forward step 3!";
}

void sync_bn_backward_step1_cpu(HostContext& ctx,
                               const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
    PARROTS_NOT_IMPL << "Not support cpu sync bn backward step 1!";
}

void sync_bn_backward_step2_cpu(HostContext& ctx,
                               const SSElement& attr,
                               const OperatorBase::in_list_t& ins,
                               OperatorBase::out_list_t& outs) {
    PARROTS_NOT_IMPL << "Not support cpu sync bn backward step 2!";
}



#ifdef PARROTS_USE_CUDA
#define CHECK_CUDA(x) PARROTS_CHECKARGS(int(x.deviceArch()) == 1) << #x << "must be a CUDA tensor"
#define CHECK_CONTIGUOUS(x) PARROTS_CHECKARGS(x.isContiguous()) << #x << "must be contoguous"
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
void cudaSyncBNForwardStep1(size_t n, size_t c, size_t h, size_t w,
                            const DArrayLite input, DArrayLite mean, cudaStream_t stream);

void cudaSyncBNForwardStep2(size_t n, size_t c, size_t h, size_t w, const DArrayLite input,
                            const DArrayLite mean, DArrayLite var, cudaStream_t stream);

void cudaSyncBNForwardStep3(size_t n, size_t c, size_t h, size_t w, size_t group_size, const DArrayLite input,
                            const float eps, const float momentum, const DArrayLite mean, const DArrayLite var,
                            DArrayLite running_mean, DArrayLite running_var, const DArrayLite weight,
                            const DArrayLite bias, DArrayLite std, DArrayLite output, cudaStream_t stream);

void cudaSyncBNBackwardStep1(size_t n, size_t c, size_t h, size_t w, const DArrayLite input,
                             const DArrayLite mean, DArrayLite weight_diff, DArrayLite bias_diff,
                             const DArrayLite std, const DArrayLite grad_output, cudaStream_t stream);

void cudaSyncBNBackwardStep2(size_t n, size_t c, size_t h, size_t w, const DArrayLite input,
                            DArrayLite grad_input, const DArrayLite mean, const DArrayLite weight,
                            const DArrayLite weight_diff, const DArrayLite bias_diff, const DArrayLite std,
                            const DArrayLite grad_output, cudaStream_t stream);


void sync_bn_forward_step1_cuda(CudaContext& ctx,
                            const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
    size_t n, c, h, w;
    SSAttrs(attr).get<size_t>("n", n)
                 .get<size_t>("c", c)
                 .get<size_t>("h", h)
                 .get<size_t>("w", w)
                 .done();

    const auto& input = ins[0];
    auto& mean = outs[0];

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    cudaSyncBNForwardStep1(n, c, h, w, input, mean, stream);
}

void sync_bn_forward_step2_cuda(CudaContext& ctx,
                            const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
    size_t n, c, h, w;
    SSAttrs(attr).get<size_t>("n", n)
                 .get<size_t>("c", c)
                 .get<size_t>("h", h)
                 .get<size_t>("w", w)
                 .done();

    const auto& input = ins[0];
    const auto& mean = ins[1];
    auto& var = outs[0];

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    cudaSyncBNForwardStep2(n, c, h, w, input, mean, var, stream);
}


void sync_bn_forward_step3_cuda(CudaContext& ctx,
                            const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
    size_t n, c, h, w, group_size;
    float eps, momentum;
    SSAttrs(attr).get<size_t>("n", n)
                 .get<size_t>("c", c)
                 .get<size_t>("h", h)
                 .get<size_t>("w", w)
                 .get<size_t>("group_size", group_size)
                 .get<float>("eps", eps)
                 .get<float>("momentum", momentum)
                 .done();

    const auto& input = ins[0];
    const auto& mean = ins[1];
    const auto& var = ins[2];
    const auto& weight = ins[3];
    const auto& bias = ins[4];
    auto& running_mean = outs[0];
    auto& running_var = outs[1];
    auto& std = outs[2];
    auto& output = outs[3];

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    cudaSyncBNForwardStep3(n, c, h, w, group_size, input, eps, momentum, mean, var, running_mean,
                           running_var, weight, bias, std, output, stream);
}

void sync_bn_backward_step1_cuda(CudaContext& ctx,
                            const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
    size_t n, c, h, w;
    SSAttrs(attr).get<size_t>("n", n)
                 .get<size_t>("c", c)
                 .get<size_t>("h", h)
                 .get<size_t>("w", w)
                 .done();

    const auto& input = ins[0];
    const auto& mean = ins[1];
    const auto& std = ins[2];
    const auto& grad_output = ins[3];
    auto& weight_diff = outs[0];
    auto& bias_diff = outs[1];

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    cudaSyncBNBackwardStep1(n, c, h, w, input, mean, weight_diff, bias_diff, std, grad_output, stream);
}

void sync_bn_backward_step2_cuda(CudaContext& ctx,
                            const SSElement& attr,
                            const OperatorBase::in_list_t& ins,
                            OperatorBase::out_list_t& outs) {
    size_t n, c, h, w;
    SSAttrs(attr).get<size_t>("n", n)
                 .get<size_t>("c", c)
                 .get<size_t>("h", h)
                 .get<size_t>("w", w)
                 .done();

    const auto& input = ins[0];
    const auto& mean = ins[1];
    const auto& weight = ins[2];
    const auto& weight_diff = ins[3];
    const auto& bias_diff = ins[4];
    const auto& std = ins[5];
    const auto& grad_output = ins[6];
    auto& grad_input = outs[0];

    cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
    cudaSyncBNBackwardStep2(n, c, h, w, input, grad_input, mean, weight, weight_diff, bias_diff,
                            std, grad_output, stream);
}

#endif


PARROTS_EXTENSION_REGISTER(syncbn_forward_step1)
    .attr("n")
    .attr("c")
    .attr("h")
    .attr("w")
    .input(1)
    .output(1)
    .apply(sync_bn_forward_step1_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(sync_bn_forward_step1_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(syncbn_forward_step2)
    .attr("n")
    .attr("c")
    .attr("h")
    .attr("w")
    .input(2)
    .output(1)
    .apply(sync_bn_forward_step2_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(sync_bn_forward_step2_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(syncbn_forward_step3)
    .attr("n")
    .attr("c")
    .attr("h")
    .attr("w")
    .attr("group_size")
    .attr("eps")
    .attr("momentum")
    .input(5)
    .output(4)
    .apply(sync_bn_forward_step3_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(sync_bn_forward_step3_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(syncbn_backward_step1)
    .attr("n")
    .attr("c")
    .attr("h")
    .attr("w")
    .input(4)
    .output(2)
    .apply(sync_bn_backward_step1_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(sync_bn_backward_step1_cuda)
#endif
    .done();

PARROTS_EXTENSION_REGISTER(syncbn_backward_step2)
    .attr("n")
    .attr("c")
    .attr("h")
    .attr("w")
    .input(7)
    .output(1)
    .apply(sync_bn_backward_step2_cpu)
#ifdef PARROTS_USE_CUDA
    .apply(sync_bn_backward_step2_cuda)
#endif
    .done();
