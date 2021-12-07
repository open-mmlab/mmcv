// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/interpolate.cpp

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void ThreeNNForwardCUDAKernelLauncher(int b, int n, int m, const Tensor unknown,
                                      const Tensor known, Tensor dist2,
                                      Tensor idx);

void three_nn_forward_cuda(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx) {
  ThreeNNForwardCUDAKernelLauncher(b, n, m, unknown, known, dist2, idx);
};
#endif

void three_nn_forward(Tensor unknown_tensor, Tensor known_tensor,
                      Tensor dist2_tensor, Tensor idx_tensor, int b, int n,
                      int m) {
  if (unknown_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    three_nn_forward_cuda(b, n, m, unknown_tensor, known_tensor, dist2_tensor,
                          idx_tensor);
#else
    AT_ERROR("three_nn is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("three_nn is not implemented on CPU");
  }
}
