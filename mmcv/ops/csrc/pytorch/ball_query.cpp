// Modified from
// https://github.com/sshaoshuai/Pointnet2.PyTorch/tree/master/pointnet2/src/ball_query.cpp

#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#ifdef MMCV_WITH_DIOPI
#include <diopi/diopirt.h>
#include <diopi/functions.h>
#include <diopi/functions_mmcv.h>

#include "diopi.hpp"
#endif

void ball_query_forward_impl(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx) {
  DISPATCH_DEVICE_IMPL(ball_query_forward_impl, b, n, m, min_radius, max_radius,
                       nsample, new_xyz, xyz, idx);
}

void ball_query_forward(Tensor new_xyz_tensor, Tensor xyz_tensor,
                        Tensor idx_tensor, int b, int n, int m,
                        float min_radius, float max_radius, int nsample) {
#ifdef MMCV_WITH_DIOPI
  auto new_xyz_tensor_ = toDiopiTensorHandle(new_xyz_tensor);
  diopiDevice_t device;
  diopiDtype_t dtype;
  diopiGetTensorDtype(new_xyz_tensor_, &dtype);
  diopiGetTensorDevice(new_xyz_tensor_, &device);
  if (device == diopi_host|| dtype == diopi_dtype_float16) {
    ball_query_forward_impl(b, n, m, min_radius, max_radius, nsample,
                          new_xyz_tensor, xyz_tensor, idx_tensor);
    return;
  }
  diopiContext ctx;
  diopiContextHandle_t ch = &ctx;
  auto xyz_tensor_ = toDiopiTensorHandle(xyz_tensor);
  auto idx_tensor_ = toDiopiTensorHandle(idx_tensor);
  if(&diopiBallQuery) {
    diopiBallQuery(ch, new_xyz_tensor_, xyz_tensor_, idx_tensor_, b, n, m, min_radius, max_radius, nsample);
  } else {
    ball_query_forward_impl(b, n, m, min_radius, max_radius, nsample,
                          new_xyz_tensor, xyz_tensor, idx_tensor);
  }
#else
  ball_query_forward_impl(b, n, m, min_radius, max_radius, nsample,
                          new_xyz_tensor, xyz_tensor, idx_tensor);
#endif
}

void stack_ball_query_forward_impl(float max_radius, int nsample,
                                   const Tensor new_xyz,
                                   const Tensor new_xyz_batch_cnt,
                                   const Tensor xyz, const Tensor xyz_batch_cnt,
                                   Tensor idx) {
  DISPATCH_DEVICE_IMPL(stack_ball_query_forward_impl, max_radius, nsample,
                       new_xyz, new_xyz_batch_cnt, xyz, xyz_batch_cnt, idx);
}

void stack_ball_query_forward(Tensor new_xyz_tensor, Tensor new_xyz_batch_cnt,
                              Tensor xyz_tensor, Tensor xyz_batch_cnt,
                              Tensor idx_tensor, float max_radius,
                              int nsample) {
  stack_ball_query_forward_impl(max_radius, nsample, new_xyz_tensor,
                                new_xyz_batch_cnt, xyz_tensor, xyz_batch_cnt,
                                idx_tensor);
}
