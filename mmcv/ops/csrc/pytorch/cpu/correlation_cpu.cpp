#include "pytorch_cpp_helper.hpp"
#include "pytorch_device_registry.hpp"
#include <torch/types.h>
#include <vector>
#include <cstdio>
#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < H && y >= 0 && y < W)
// template <typename scalar_t>
// static void correlate_patch(
//     Tensor input1,
//     Tensor input2,
//     scalar_t *dst,
//     int kH, int kW,
//     int dilationH, int dilationW,
//     int u, int v,
//     int shiftU, int shiftV)
// {
//   const int C = input1.size(0);
//   const int iH = input1.size(1);
//   const int iW = input1.size(2);
//   for (int c = 0; c < C; ++c)
//   {
//     for (int i = 0; i < kH; ++i)
//     {
//       int i1 = u + i * dilationH;
//       int i2 = i1 + shiftU;
//       if WITHIN_BOUNDS (i1, i2, iH, iH)
//       {
//         for (int j = 0; j < kW; ++j)
//         {
//           int j1 = v + j * dilationW;
//           int j2 = j1 + shiftV;
//           if WITHIN_BOUNDS (j1, j2, iW, iW)
//           {
//             scalar_t v1 = input1[c][i1][j1];
//             scalar_t v2 = input2[c][i2][j2];
//             *dst += v1 * v2;
//           }
//         }
//       }
//     }
//   }
// }
template <typename scalar_t>
static void correlate_forward_cpu_kernel(
    TensorAccessor<scalar_t, 3> input1,
    TensorAccessor<scalar_t, 3> input2,
    scalar_t *dst,
    int kH, int kW,
    int dilationH, int dilationW,
    int u, int v,
    int shiftU, int shiftV)
{
  const int C = input1.size(0);
  const int iH = input1.size(1);
  const int iW = input1.size(2);
  for (int c = 0; c < C; ++c)
  {
    for (int i = 0; i < kH; ++i)
    {
      int i1 = u + i * dilationH;
      int i2 = i1 + shiftU;
      if WITHIN_BOUNDS (i1, i2, iH, iH)
      {
        for (int j = 0; j < kW; ++j)
        {
          int j1 = v + j * dilationW;
          int j2 = j1 + shiftV;
          if WITHIN_BOUNDS (j1, j2, iW, iW)
          {
            scalar_t v1 = input1[c][i1][j1];
            scalar_t v2 = input2[c][i2][j2];
            *dst += v1 * v2;
          }
        }
      }
    }
  }
}

template <typename scalar_t>
static void correlation_backward_cpu_kernel(
    TensorAccessor<scalar_t, 3> input1,
    TensorAccessor<scalar_t, 3> gradInput1,
    TensorAccessor<scalar_t, 3> input2,
    TensorAccessor<scalar_t, 3> gradInput2,
    scalar_t &gradOutput,
    int kH, int kW,
    int dilationH, int dilationW,
    int u, int v,
    int shiftU, int shiftV)
{

  const int C = input1.size(0);
  const int iH = input1.size(1);
  const int iW = input1.size(2);

  for (int c = 0; c < C; ++c)
  {
    for (int i = 0; i < kH; ++i)
    {
      int i1 = u + i * dilationH;
      int i2 = i1 + shiftU;
      if WITHIN_BOUNDS (i1, i2, iH, iH)
      {
        for (int j = 0; j < kW; ++j)
        {
          int j1 = v + j * dilationW;
          int j2 = j1 + shiftV;
          if WITHIN_BOUNDS (j1, j2, iW, iW)
          {
            scalar_t v1 = input1[c][i1][j1];
            scalar_t v2 = input2[c][i2][j2];
            gradInput2[c][i2][j2] += gradOutput * v1;
            gradInput1[c][i1][j1] += gradOutput * v2;
          }
        }
      }
    }
  }
}

void correlationForwardCPUKernelLaucher(const Tensor input1,
                                        const Tensor input2,
                                        Tensor output,
                                        int kH, int kW,
                                        int patchH, int patchW,
                                        int padH, int padW,
                                        int dilationH, int dilationW,
                                        int dilation_patchH, int dilation_patchW,
                                        int dH, int dW)
{
  const auto batch_size = input1.size(0);
  const auto iH = input1.size(2);
  const auto iW = input1.size(3);
  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;
  const int dilatedKH = (kH - 1) * dilationH + 1;
  const int dilatedKW = (kW - 1) * dilationW + 1;

  const auto oH = (iH + 2 * padH - dilatedKH) / dH + 1;
  const auto oW = (iW + 2 * padW - dilatedKW) / dW + 1;
  // auto output = at::zeros({batch_size, patchH, patchW, oH, oW}, input1.options());

  int n, ph, pw, h, w;
#pragma omp parallel for private(n, ph, pw, h, w) collapse(2)
  for (n = 0; n < batch_size; ++n)
  {
    for (ph = 0; ph < patchH; ++ph)
    {
      for (pw = 0; pw < patchW; ++pw)
      {
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input1.scalar_type(), "correlation_forward_cpp", ([&]
                                                                                              {
            auto input1_acc = input1.accessor<scalar_t, 4>();
            auto input2_acc = input2.accessor<scalar_t, 4>();
            auto output_acc = output.accessor<scalar_t, 5>();
            for (h = 0; h < oH; ++h) {
              for (w = 0; w < oW; ++w) {
                correlate_forward_cpu_kernel(input1_acc[n],
                                input2_acc[n],
                                &output_acc[n][ph][pw][h][w],
                                kH, kW,
                                dilationH, dilationW,
                                -padH + h * dH,
                                -padW + w * dW,
                                (ph - patchRadH)  * dilation_patchH,
                                (pw - patchRadW)  * dilation_patchW);
              }
            } }));
      }
    }
  }
  // return output;
}
void correlationBackwardCPUKernelLaucher(const torch::Tensor gradOutput,
                                         torch::Tensor input1,
                                         torch::Tensor input2,
                                         Tensor grad_input1,
                                         Tensor grad_input2,
                                         int kH, int kW,
                                         int patchH, int patchW,
                                         int padH, int padW,
                                         int dilationH, int dilationW,
                                         int dilation_patchH, int dilation_patchW,
                                         int dH, int dW)
{

  const int batch_size = input1.size(0);
  const int patchRadH = (patchH - 1) / 2;
  const int patchRadW = (patchW - 1) / 2;
  const int oH = gradOutput.size(3);
  const int oW = gradOutput.size(4);

  int n, ph, pw, h, w;
#pragma omp parallel for private(n, ph, pw, h, w)
  for (n = 0; n < batch_size; ++n)
  {
    AT_DISPATCH_FLOATING_TYPES(input1.scalar_type(), "correlation_backward_cpp", ([&]
                                                                                  {
        auto input1_acc = input1.accessor<scalar_t, 4>();
        auto gradInput1_acc = grad_input1.accessor<scalar_t, 4>();
        auto input2_acc = input2.accessor<scalar_t, 4>();
        auto gradInput2_acc = grad_input2.accessor<scalar_t, 4>();
        auto gradOutput_acc = gradOutput.accessor<scalar_t, 5>();

        for(ph = 0; ph < patchH; ++ph){
          for(pw = 0; pw < patchW; ++pw){
            for (h = 0; h < oH; ++h) {
              for (w = 0; w < oW; ++w) {
                correlation_backward_cpu_kernel(input1_acc[n], gradInput1_acc[n],
                                     input2_acc[n], gradInput2_acc[n],
                                     gradOutput_acc[n][ph][pw][h][w],
                                     kH, kW,
                                     dilationH, dilationW,
                                     -padH + h * dH,
                                     -padW + w * dW,
                                     (ph - patchRadH)  * dilation_patchH,
                                     (pw - patchRadW)  * dilation_patchW);
              }
            }
          }
        } }));
  }

  // return {gradInput1, gradInput2};
}

void correlation_forward_cpu(Tensor input1, Tensor input2, Tensor output,
                             int kH, int kW, int patchH, int patchW, int padH,
                             int padW, int dilationH, int dilationW,
                             int dilation_patchH, int dilation_patchW, int dH,
                             int dW)
{
  correlationForwardCPUKernelLaucher(input1, input2, output,
                                     kH, kW, patchH, patchW, padH,
                                     padW, dilationH, dilationW,
                                     dilation_patchH, dilation_patchW, dH,
                                     dW);
}
void correlation_backward_cpu(Tensor gradOutput,
                              Tensor input1,
                              Tensor input2,
                              Tensor grad_input1,
                              Tensor grad_input2,
                              int kH, int kW,
                              int patchH, int patchW,
                              int padH, int padW,
                              int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW,
                              int dH, int dW)
{
  correlationBackwardCPUKernelLaucher(input1, input2,
                                      gradOutput,
                                      grad_input1,
                                      grad_input2,
                                      kH, kW,
                                      patchH, patchW,
                                      padH, padW,
                                      dilationH, dilationW,
                                      dilation_patchH, dilation_patchW,
                                      dH, dW);
}
void correlation_forward_impl(Tensor input1, Tensor input2, Tensor output,
                              int kH, int kW, int patchH, int patchW, int padH,
                              int padW, int dilationH, int dilationW,
                              int dilation_patchH, int dilation_patchW, int dH,
                              int dW);
void correlation_backward_impl(Tensor gradOutput,
                               Tensor input1,
                               Tensor input2,
                               Tensor grad_input1,
                               Tensor grad_input2,
                               int kH, int kW,
                               int patchH, int patchW,
                               int padH, int padW,
                               int dilationH, int dilationW,
                               int dilation_patchH, int dilation_patchW,
                               int dH, int dW);
REGISTER_DEVICE_IMPL(correlation_forward_impl, CPU, correlation_forward_cpu);
REGISTER_DEVICE_IMPL(correlation_backward_impl, CPU, correlation_backward_cpu);
