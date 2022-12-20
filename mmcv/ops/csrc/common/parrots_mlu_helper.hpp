#ifndef PARROTS_MLU_HELPER
#define PARROTS_MLU_HELPER

#include <parrots/extension.hpp>

#ifdef PARROTS_USE_CAMB

#include <cnrt.h>

#ifdef __BANG_ARCH__
#define MAX_NRAM_SIZE \
  (__MLU_NRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#define MAX_SRAM_SIZE \
  (__MLU_SRAM_SIZE__ * 1024 - REM_FOR_STACK)  // 128KB reserved for cncc
#else
#define MAX_NRAM_SIZE (384 * 1024)   // 384KB,  initialization value
#define MAX_SRAM_SIZE (1920 * 1024)  // 1920KB, initialization value
#endif

#define NFU_ALIGN_SIZE 128
#define PAD_UP(x, y) (((x) / (y) + (int)((x) % (y) > 0)) * (y))
#define PAD_DOWN(x, y) (((x) / (y)) * (y))
#define CEIL_ALIGN(x, y) (((x) + (y)-1) / (y) * (y))

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

constexpr uint32_t rem_for_stack = 128 * 1024;

inline uint32_t getDeviceAttr(cnrtDeviceAttr_t attr) {
  int dev_ordinal = 0;
  int device_attr = 1;
  cnrtGetDevice(&dev_ordinal);
  cnrtDeviceGetAttribute(&device_attr, attr, dev_ordinal);
  if (attr == cnrtAttrNramSizePerMcore) {
    device_attr -= rem_for_stack;
  }
  return device_attr;
}

#endif  // PARROTS_USE_CAMB

#endif  // !PARROTS_MLU_HELPER
