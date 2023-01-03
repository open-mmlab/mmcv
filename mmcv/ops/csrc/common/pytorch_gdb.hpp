#ifndef PYTORCH_GDB_HPP
#define PYTORCH_GDB_HPP

namespace torch {
namespace gdb {
/* ~~~ misc debugging utilities ~~~
 *
 * torch::gdb::* functions are NOT meant to be called by general pytorch code,
 * but only from within a gdb session. As such, utils.h does not contain any
 * declaration for those.
 */

// This is a helper needed by the torch-tensor-repr gdb command.
// Return an human-readable representation of the given Tensor. The resulting
// string is stored into a malloc()ed buffer. The caller is responsible to
// free() it. We use malloc() instead of new[] because it's much easier to
// call free than delete[] from within gdb.
// Currently the code for computing the repr of a tensor is written in Python,
// so we need to wrap the Tensor into a Python object first.
char *tensor_repr(at::Tensor tensor);
} // namespace gdb
} // namespace torch

#endif // PYTORCH_GDB_HPP
