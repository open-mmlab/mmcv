#include "parrots_cpp_helper.hpp"

void MaskedIm2colForwardCUDAKernelLauncher(
    const DArrayLite bottom_data, const DArrayLite mask_h_idx,
    const DArrayLite mask_w_idx, DArrayLite top_data, const int kernel_h,
    const int kernel_w, const int pad_h, const int pad_w, cudaStream_t stream);

void MaskedCol2imForwardCUDAKernelLaucher(const DArrayLite bottom_data,
                                          const DArrayLite mask_h_idx,
                                          const DArrayLite mask_w_idx,
                                          DArrayLite top_data, const int height,
                                          const int width, const int channels,
                                          cudaStream_t stream);

void masked_im2col_forward_cuda(CudaContext& ctx, const SSElement& attr,
                                const OperatorBase::in_list_t& ins,
                                OperatorBase::out_list_t& outs) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  int kernel_h, kernel_w, pad_h, pad_w;
  SSAttrs(attr)
      .get<int>("kernel_h", kernel_h)
      .get<int>("kernel_w", kernel_w)
      .get<int>("pad_h", pad_h)
      .get<int>("pad_w", pad_w)
      .done();

  const auto& im = ins[0];
  const auto& mask_h_idx = ins[1];
  const auto& mask_w_idx = ins[2];

  auto& col = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  MaskedIm2colForwardCUDAKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                        kernel_h, kernel_w, pad_h, pad_w,
                                        stream);
}

void masked_col2im_forward_cuda(CudaContext& ctx, const SSElement& attr,
                                const OperatorBase::in_list_t& ins,
                                OperatorBase::out_list_t& outs) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  int height, width, channels;
  SSAttrs(attr)
      .get<int>("height", height)
      .get<int>("width", width)
      .get<int>("channels", channels)
      .done();

  const auto& col = ins[0];
  const auto& mask_h_idx = ins[1];
  const auto& mask_w_idx = ins[2];

  auto& im = outs[0];

  cudaStream_t stream = getStreamNative<CudaDevice>(ctx.getStream());
  MaskedCol2imForwardCUDAKernelLaucher(col, mask_h_idx, mask_w_idx, im, height,
                                       width, channels, stream);
}

PARROTS_EXTENSION_REGISTER(masked_im2col_forward)
    .attr("kernel_h")
    .attr("kernel_w")
    .attr("pad_h")
    .attr("pad_w")
    .input(3)
    .output(1)
    .apply(masked_im2col_forward_cuda)
    .done();

PARROTS_EXTENSION_REGISTER(masked_col2im_forward)
    .attr("height")
    .attr("width")
    .attr("channels")
    .input(3)
    .output(1)
    .apply(masked_col2im_forward_cuda)
    .done();
