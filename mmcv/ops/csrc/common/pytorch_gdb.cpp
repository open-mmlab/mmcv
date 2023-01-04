#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/python_headers.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/python_strings.h>
#include <torch/csrc/utils/python_tuples.h>

#include <torch/csrc/Export.h>

#include <algorithm>
#include <cstdarg>
#include <iterator>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <pytorch_gdb.hpp>

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
char* tensor_repr(at::Tensor tensor) {
  PyGILState_STATE gil = PyGILState_Ensure();
  PyObject* pytensor = nullptr;
  PyObject* repr = nullptr;
  // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
  Py_ssize_t bufsize;
  const char* buf = nullptr;
  char* result = nullptr;

  pytensor = THPVariable_Wrap(at::Tensor(tensor));
  if (!pytensor)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  repr = PyObject_Repr(pytensor);
  if (!repr)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  buf = PyUnicode_AsUTF8AndSize(repr, &bufsize);
  if (!buf)
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  result =
      static_cast<char*>(malloc(bufsize + 1)); // account for the trailing \0
  if (!result) {
    fprintf(stderr, "cannot allocate memory for the result\n");
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto,hicpp-avoid-goto)
    goto error;
  }
  // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.strcpy)
  strcpy(result, buf);
  Py_XDECREF(pytensor);
  Py_XDECREF(repr);
  PyGILState_Release(gil);
  return result;

error:
  fprintf(stderr, "torch::gdb::tensor_repr: unexpected error\n");
  if (PyErr_Occurred())
    PyErr_Print();
  Py_XDECREF(pytensor);
  Py_XDECREF(repr);
  // NOLINTNEXTLINE(cppcoreguidelines-no-malloc)
  free(result);
  PyGILState_Release(gil);
  return nullptr;
}

} // namespace gdb
} // namespace torch
