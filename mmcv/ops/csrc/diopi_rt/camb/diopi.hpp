#ifndef INCLUDE_DIOPI_HPP_
#define INCLUDE_DIOPI_HPP_

#include <diopi/diopirt.h>
#include <list>

struct diopiContext {
  std::list<at::Tensor> arrays;
  diopiContext() {}
};

diopiTensorHandle_t toDiopiTensorHandle(at::Tensor& tensor);
diopiConstTensorHandle_t toDiopiTensorHandle(const at::Tensor& tensor);
diopiTensorHandle_t toDiopiTensorHandleWithConstCase(const at::Tensor& tensor);

#endif // INCLUDE_PARROTS_DIOPI_HPP_
