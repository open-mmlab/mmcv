#ifndef _GATHER_POINT_EXT_PYTORCH
#define _GATHER_POINT_EXT_PYTORCH

#include <torch/extension.h>
using namespace at;

int gather_points(int b, int c, int n, int npoints,
                  Tensor points, Tensor idx,
                  Tensor out);

int gather_points_backward(int b, int c, int n, int npoints, 
                           Tensor grad_out, Tensor idx,
                           Tensor grad_points);
#endif // _GATHER_POINT_EXT_PYTORCH
