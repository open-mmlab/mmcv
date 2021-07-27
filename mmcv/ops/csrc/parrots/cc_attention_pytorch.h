#ifndef CC_ATTENTION_PYTORCH_H
#define CC_ATTENTION_PYTORCH_H
#include <torch/extension.h>
using namespace at;

void ca_forward_cuda(const Tensor t, const Tensor f, Tensor weight);

void ca_backward_cuda(const Tensor dw, const Tensor t, const Tensor f,
                      Tensor dt, Tensor df);

void ca_map_forward_cuda(const Tensor weight, const Tensor g, Tensor out);

void ca_map_backward_cuda(const Tensor dout, const Tensor weight,
                          const Tensor g, Tensor dw, Tensor dg);
#endif  // CC_ATTENTION_PYTORCH_H
