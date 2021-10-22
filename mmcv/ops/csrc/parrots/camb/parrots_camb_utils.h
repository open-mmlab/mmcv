// Copyright (c) 2021, SenseTime.

#ifndef PARROTS_CAMB_UTILS
#define PARROTS_CAMB_UTILS

#include <parrots/extension.hpp>

#ifdef PARROTS_USE_CAMB

#include <cnrt.h>

#include "mlu_utils.h"

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

inline int getDeviceAttr(const cnrtDeviceAttr_t& attr) {
  int ordinal = -1;
  cnrtGetDevice(&ordinal);
  int value = 0;
  cnrtDeviceGetAttribute(&value, attr, ordinal);
  return value;
}

inline int itemsize(const DArrayLite& x) { return x.nbytes() / x.size(); }

#endif  // PARROTS_USE_CAMB

#endif  // !PARROTS_CAMB_UTILS