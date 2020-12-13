#ifndef _INTERPOLATE_EXT_PYTORCH
#define _INTERPOLATE_EXT_PYTORCH
#include <torch/extension.h>
using namespace at;

void three_nn(int b, int n, int m, const Tensor unknown, const Tensor known,
              Tensor dist2, Tensor idx);

void three_interpolate(int b, int c, int m, int n, const Tensor points,
                       const Tensor idx, const Tensor weight, Tensor out);

void three_interpolate_backward(int b, int c, int n, int m,
                                const Tensor grad_out, const Tensor idx,
                                const Tensor weight, Tensor grad_points);
#endif //_INTERPOLATE_EXT_PYTORCH
