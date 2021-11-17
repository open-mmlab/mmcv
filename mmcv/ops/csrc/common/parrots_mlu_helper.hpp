#ifndef PARROTS_MLU_HELPER
#define PARROTS_MLU_HELPER

#include <parrots/extension.hpp>

#ifdef PARROTS_USE_CAMB

#include <cnrt.h>

#include "common_mlu_helper.hpp"

#ifndef CAMB_BENCHMARK_OP()
#define CAMB_BENCHMARK_OP() \
  {}
#endif  // !CAMB_BENCHMARK_OP

using parrots::DArrayLite;
using parrots::Prim;

inline cnrtDataType_t getCnrtDataType(parrots::ValueType vt) {
  switch (vt.prim()) {
    case Prim::Float16:
      return cnrtFloat16;
    case Prim::Float32:
      return cnrtFloat32;
    case Prim::Int16:
      return cnrtInt16;
    case Prim::Int32:
      return cnrtInt32;
    case Prim::Int64:
      return cnrtInt64;
    case Prim::Int8:
      return cnrtInt8;
    case Prim::Uint8:
      return cnrtUInt8;
    case Prim::Bool:
      return cnrtBool;
    default:
      PARROTS_NOTSUPPORTED << "Unsupported data type for CNRT: " << vt.name();
  }
}

inline int itemsize(parrots::ValueType vt) {
  switch (vt.prim()) {
    case Prim::Float16:
      return 2;
    case Prim::Float32:
      return 4;
    case Prim::Int16:
      return 2;
    case Prim::Int32:
      return 4;
    case Prim::Int64:
      return 8;
    case Prim::Int8:
      return 1;
    case Prim::Uint8:
      return 1;
    case Prim::Bool:
      return 1;
    default:
      PARROTS_NOTSUPPORTED << "Unsupported data type for CNRT: " << vt.name();
  }
}

inline int itemsize(const DArrayLite& x) { return itemsize(x.elemType()); }

inline int getDeviceAttr(const cnrtDeviceAttr_t& attr) {
  int ordinal = -1;
  cnrtGetDevice(&ordinal);
  int value = 0;
  cnrtDeviceGetAttribute(&value, attr, ordinal);
  return value;
}

#endif  // PARROTS_USE_CAMB

#endif  // !PARROTS_MLU_HELPER
