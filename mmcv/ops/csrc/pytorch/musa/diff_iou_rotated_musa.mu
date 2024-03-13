// Copyright (c) OpenMMLab. All rights reserved
// Adapted from
// https://github.com/lilanxiao/Rotated_IoU/musa_op/sort_vert_kernel.cu  # noqa
#include "diff_iou_rotated_musa_kernel.muh"
#include "pytorch_cpp_helper.hpp"
#include "pytorch_musa_helper.hpp"

at::Tensor DiffIoURotatedSortVerticesMUSAKernelLauncher(at::Tensor vertices,
                                                        at::Tensor mask,
                                                        at::Tensor num_valid) {
  c10::musa::MUSAGuard device_guard(vertices.device());
  musaStream_t stream = c10::musa::getCurrentMUSAStream();

  CHECK_CONTIGUOUS(vertices);
  CHECK_CONTIGUOUS(mask);
  CHECK_CONTIGUOUS(num_valid);
  CHECK_MUSA(vertices);
  CHECK_MUSA(mask);
  CHECK_MUSA(num_valid);

  int b = vertices.size(0);
  int n = vertices.size(1);
  int m = vertices.size(2);
  at::Tensor idx =
      torch::zeros({b, n, MAX_NUM_VERT_IDX},
                   at::device(vertices.device()).dtype(at::ScalarType::Int));

  diff_iou_rotated_sort_vertices_forward_musa_kernel<<<b, opt_n_thread(n), 0,
                                                       stream>>>(
      b, n, m, vertices.data_ptr<float>(), mask.data_ptr<bool>(),
      num_valid.data_ptr<int>(), idx.data_ptr<int>());
  AT_MUSA_CHECK(musaGetLastError());

  return idx;
}
