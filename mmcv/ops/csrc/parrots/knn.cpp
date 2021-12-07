// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap

#include "pytorch_cpp_helper.hpp"

#ifdef MMCV_WITH_CUDA
void KNNForwardCUDAKernelLauncher(int b, int n, int m, int nsample,
                                  const Tensor xyz, const Tensor new_xyz,
                                  Tensor idx, Tensor dist2);

void knn_forward_cuda(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2) {
  KNNForwardCUDAKernelLauncher(b, n, m, nsample, xyz, new_xyz, idx, dist2);
}
#endif

void knn_forward(Tensor xyz_tensor, Tensor new_xyz_tensor, Tensor idx_tensor,
                 Tensor dist2_tensor, int b, int n, int m, int nsample) {
  if (new_xyz_tensor.device().is_cuda()) {
#ifdef MMCV_WITH_CUDA
    CHECK_CUDA_INPUT(new_xyz_tensor);
    CHECK_CUDA_INPUT(xyz_tensor);

    knn_forward_cuda(b, n, m, nsample, xyz_tensor, new_xyz_tensor, idx_tensor,
                     dist2_tensor);
#else
    AT_ERROR("knn is not compiled with GPU support");
#endif
  } else {
    AT_ERROR("knn is not implemented on CPU");
  }
}
