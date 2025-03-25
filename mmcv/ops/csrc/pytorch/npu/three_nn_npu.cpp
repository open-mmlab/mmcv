#include "pytorch_npu_helper.hpp"
#include "torch_npu/csrc/aten/NPUNativeFunctions.h"
#include "torch_npu/csrc/framework/utils/OpAdapter.h"

using namespace NPU_NAME_SPACE;
using namespace std;

void three_nn_forward_npu(int b, int n, int m, const Tensor unknown,
                          const Tensor known, Tensor dist2, Tensor idx) {
  at::Tensor source = known.contiguous();
  at::Tensor target = unknown.contiguous();

  bool is_from_knn = false;
  int nsample = 3;
  EXEC_NPU_CMD(aclnnKnn, source, target, is_from_knn, nsample, dist2, idx);
}

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx);

REGISTER_NPU_IMPL(three_nn_forward_impl, three_nn_forward_npu);
