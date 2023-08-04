/*************************************************************************
 * Copyright (C) 2022 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include "mlu_common_helper.h"

void ThreeNNMLUKernelLauncher(int b, int n, int m, const Tensor unknown,
                              const Tensor known, Tensor dist2, Tensor idx) {
  auto unknown_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      unknown, unknown.suggest_memory_format());
  auto known_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      known, known.suggest_memory_format());
  auto dist2_contiguous = torch_mlu::cnnl::ops::cnnl_contiguous(
      dist2, dist2.suggest_memory_format());
  auto idx_contiguous =
      torch_mlu::cnnl::ops::cnnl_contiguous(idx, idx.suggest_memory_format());

  MluOpTensorDescriptor unknown_desc, known_desc, dist2_desc, idx_desc;
  unknown_desc.set(unknown_contiguous);
  known_desc.set(known_contiguous);
  dist2_desc.set(dist2_contiguous);
  idx_desc.set(idx_contiguous);

  auto handle = mluOpGetCurrentHandle();
  size_t workspace_size = 0;
  TORCH_MLUOP_CHECK(mluOpGetThreeNNForwardWorkspaceSize(
      handle, known_desc.desc(), &workspace_size));
  auto known_workspace =
      at::empty(workspace_size, known.options().dtype(at::kByte));

  auto unknown_impl = torch_mlu::getMluTensorImpl(unknown_contiguous);
  auto known_impl = torch_mlu::getMluTensorImpl(known_contiguous);
  auto dist2_impl = torch_mlu::getMluTensorImpl(dist2_contiguous);
  auto idx_impl = torch_mlu::getMluTensorImpl(idx_contiguous);
  auto workspace_impl = torch_mlu::getMluTensorImpl(known_workspace);
  auto unknown_ptr = unknown_impl->cnnlMalloc();
  auto known_ptr = known_impl->cnnlMalloc();
  auto dist2_ptr = dist2_impl->cnnlMalloc();
  auto idx_ptr = idx_impl->cnnlMalloc();
  auto workspace_ptr = workspace_impl->cnnlMalloc();

  TORCH_MLUOP_CHECK(mluOpThreeNNForward(
      handle, unknown_desc.desc(), unknown_ptr, known_desc.desc(), known_ptr,
      workspace_ptr, workspace_size, dist2_desc.desc(), dist2_ptr,
      idx_desc.desc(), idx_ptr));
}

void three_nn_forward_mlu(int b, int n, int m, const Tensor unknown,
                          const Tensor known, Tensor dist2, Tensor idx) {
  ThreeNNMLUKernelLauncher(b, n, m, unknown, known, dist2, idx);
}

void three_nn_forward_impl(int b, int n, int m, const Tensor unknown,
                           const Tensor known, Tensor dist2, Tensor idx);

REGISTER_DEVICE_IMPL(three_nn_forward_impl, MLU, three_nn_forward_mlu);
