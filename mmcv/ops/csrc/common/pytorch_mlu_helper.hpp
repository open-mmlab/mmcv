/*************************************************************************
 * Copyright (C) 2021 Cambricon.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#ifndef PYTORCH_MLU_HELPER_HPP_
#define PYTORCH_MLU_HELPER_HPP_

#ifdef MMCV_WITH_MLU
#include "aten.h"

#define NFU_ALIGN_SIZE 128

#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))

#define PAD_DOWN(x, y) (((x) / (y)) * (y))

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

#define CEIL_ALIGN(x, y) (((x) + (y)-1) / (y) * (y))

inline int32_t getJobLimitCapability() {
  CNcontext drv_ctx;
  TORCH_CHECK(CN_SUCCESS == cnCtxGetCurrent(&drv_ctx), "cnCtxGetCurrent fails");
  CNctxConfigParam ctx_conf_param;
  TORCH_CHECK(
      CN_SUCCESS == cnGetCtxConfigParam(drv_ctx, CN_CTX_CONFIG_UNION_LIMIT,
                                        &ctx_conf_param),
      "cnGetCtxConfigParam fails.");
  return (int32_t)ctx_conf_param.unionLimit;
}

#endif  // MMCV_WITH_MLU

#endif  // PYTORCH_MLU_HELPER_HPP_
