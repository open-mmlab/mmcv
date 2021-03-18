#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void MaskedIm2colForwardCUDAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int kernel_h,
                                           const int kernel_w, const int pad_h,
                                           const int pad_w);

void MaskedCol2imForwardCUDAKernelLauncher(const Tensor bottom_data,
                                           const Tensor mask_h_idx,
                                           const Tensor mask_w_idx,
                                           Tensor top_data, const int height,
                                           const int width, const int channels);

void masked_im2col_forward_cuda(const Tensor im, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor col,
                                const int kernel_h, const int kernel_w,
                                const int pad_h, const int pad_w) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kw), col: (kh * kw * ic, ow * oh)
  MaskedIm2colForwardCUDAKernelLauncher(im, mask_h_idx, mask_w_idx, col,
                                        kernel_h, kernel_w, pad_h, pad_w);
}

void masked_col2im_forward_cuda(const Tensor col, const Tensor mask_h_idx,
                                const Tensor mask_w_idx, Tensor im, int height,
                                int width, int channels) {
  // im: (n, ic, h, w), kernel size (kh, kw)
  // kernel: (oc, ic * kh * kh), col: (kh * kw * ic, ow * oh)
  MaskedCol2imForwardCUDAKernelLauncher(col, mask_h_idx, mask_w_idx, im, height,
                                        width, channels);
}
#endif

void masked_im2col_forward(const Tensor im, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor col,
                           const int kernel_h, const int kernel_w,
                           const int pad_h, const int pad_w) {
  if (im.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(im);
    CHECK_CUDA_INPUT(mask_h_idx);
    CHECK_CUDA_INPUT(mask_w_idx);
    CHECK_CUDA_INPUT(col);
    masked_im2col_forward_cuda(im, mask_h_idx, mask_w_idx, col, kernel_h,
                               kernel_w, pad_h, pad_w);
#else
    AT_ERROR("MaskConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("MaskConv is not implemented on CPU");
  }
}

void masked_col2im_forward(const Tensor col, const Tensor mask_h_idx,
                           const Tensor mask_w_idx, Tensor im, int height,
                           int width, int channels) {
  if (col.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(col);
    CHECK_CUDA_INPUT(mask_h_idx);
    CHECK_CUDA_INPUT(mask_w_idx);
    CHECK_CUDA_INPUT(im);
    masked_col2im_forward_cuda(col, mask_h_idx, mask_w_idx, im, height, width,
                               channels);
#else
    AT_ERROR("MaskConv is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("MaskConv is not implemented on CPU");
  }
}
