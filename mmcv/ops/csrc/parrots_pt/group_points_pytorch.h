#ifndef _GROUP_POINTS_EXT_PYTORCH
#define _GROUP_POINTS_EXT_PYTORCH

#include <torch/extension.h>
using namespace at;

int group_points(int b, int c, int n, int npoints, int nsample, Tensor points,
                 Tensor idx, Tensor out);

int group_points_backward(int b, int c, int n, int npoints, int nsample,
                          Tensor grad_out, Tensor idx, Tensor grad_points);

#endif //_GROUP_POINTS_EXT_PYTORCH
