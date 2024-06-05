#include "pytorch_npu_helper.hpp"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

using namespace NPU_NAME_SPACE;
using namespace std;

void knn_forward_npu(int b, int n, int m, int nsample, const Tensor xyz,
                     const Tensor new_xyz, Tensor idx, Tensor dist2) {
  // transpose known from [B, N, 3] to [B, 3, N]
  at::Tensor source = xyz.transpose(1, 2).contiguous();
  at::Tensor target = new_xyz.contiguous();

  bool is_from_knn = true;
  EXEC_NPU_CMD(aclnnKnn, source, target, nsample, is_from_knn, idx, dist2);
}

void knn_forward_impl(int b, int n, int m, int nsample, const Tensor xyz,
                      const Tensor new_xyz, Tensor idx, Tensor dist2);

REGISTER_NPU_IMPL(knn_forward_impl, knn_forward_npu);
