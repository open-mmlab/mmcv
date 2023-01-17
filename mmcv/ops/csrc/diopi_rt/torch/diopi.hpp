#ifndef INCLUDE_DIOPI_HPP_
#define INCLUDE_DIOPI_HPP_

#include <diopi/diopirt.h>
#include <list>

struct diopiContext {
  std::list<at::Tensor> arrays;
  diopiContext() {}
};

#endif // INCLUDE_PARROTS_DIOPI_HPP_
