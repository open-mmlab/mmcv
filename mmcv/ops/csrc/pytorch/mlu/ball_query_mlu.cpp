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
#include "mlu_common_desc.h"
#include "mlu_op.h"
#include "pytorch_device_registry.hpp"
#include "pytorch_mlu_helper.hpp"

void ball_query_forward_mlu(int b, int n, int m, float min_radius,
                            float max_radius, int nsample, const Tensor new_xyz,
                            const Tensor xyz, Tensor idx) {
  MluOpTensorDescriptor new_xyz_desc, xyz_desc, idx_desc;
  new_xyz_desc.set(new_xyz);
  xyz_desc.set(xyz);
  idx_desc.set(idx);

  auto new_xyz_impl = torch_mlu::getMluTensorImpl(new_xyz);
  auto xyz_impl = torch_mlu::getMluTensorImpl(xyz);
  auto idx_impl = torch_mlu::getMluTensorImpl(idx);
  auto new_xyz_ptr = new_xyz_impl->cnnlMalloc();
  auto xyz_ptr = xyz_impl->cnnlMalloc();
  auto idx_ptr = idx_impl->cnnlMalloc();

  auto handle = mluOpGetCurrentHandle();
  mluOpBallQuery(handle, new_xyz_desc.desc(), new_xyz_ptr, xyz_desc.desc(),
                 xyz_ptr, min_radius, max_radius, nsample, idx_desc.desc(),
                 idx_ptr);
}

void ball_query_forward_impl(int b, int n, int m, float min_radius,
                             float max_radius, int nsample,
                             const Tensor new_xyz, const Tensor xyz,
                             Tensor idx);

REGISTER_DEVICE_IMPL(ball_query_forward_impl, MLU, ball_query_forward_mlu);
