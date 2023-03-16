// Modified from
// https://github.com/CVMI-Lab/PAConv/tree/main/scene_seg/lib/pointops/src/knnquery_heap

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>
#include "diopi.hpp"
#endif

void knn_forward_impl(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2) {
  DISPATCH_DEVICE_IMPL(knn_forward_impl, b, n, m, nsample, xyz, new_xyz, idx,
                       dist2);
}

void knn_forward(Tensor xyz_tensor, Tensor new_xyz_tensor, Tensor idx_tensor,
                 Tensor dist2_tensor, int b, int n, int m, int nsample) {
#ifdef MMCV_WITH_DIOPI
  auto xyz_tensor_p = reinterpret_cast<diopiTensorHandle_t>(&xyz_tensor);
  diopiDevice_t device;
  diopiGetTensorDevice(xyz_tensor_p, &device);
  if (device == diopi_host) {
      knn_forward_impl(b, n, m, nsample, xyz_tensor, new_xyz_tensor, idx_tensor,
                   dist2_tensor);
      return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto new_xyz_tensor_p = reinterpret_cast<diopiTensorHandle_t>(&new_xyz_tensor);
  auto idx_tensor_p = reinterpret_cast<diopiTensorHandle_t>(&idx_tensor);
  auto dist2_tensor_p = reinterpret_cast<diopiTensorHandle_t>(&dist2_tensor);
  diopiKnn(ch, xyz_tensor_p, new_xyz_tensor_p, idx_tensor_p, dist2_tensor_p, b, n, m, nsample);
#else
  knn_forward_impl(b, n, m, nsample, xyz_tensor, new_xyz_tensor, idx_tensor,
                   dist2_tensor);
#endif
}
