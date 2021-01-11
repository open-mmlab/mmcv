#ifndef TRT_PLUGIN_HELPER_HPP
#define TRT_PLUGIN_HELPER_HPP
#include <stdexcept>

#include "NvInferPlugin.h"

namespace mmcv {

inline unsigned int getElementSize(nvinfer1::DataType t) {
  switch (t) {
    case nvinfer1::DataType::kINT32:
      return 4;
    case nvinfer1::DataType::kFLOAT:
      return 4;
    case nvinfer1::DataType::kHALF:
      return 2;
    // case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kINT8:
      return 1;
    default:
      throw std::runtime_error("Invalid DataType.");
  }
  throw std::runtime_error("Invalid DataType.");
  return 0;
}
}  // namespace mmcv
#endif  // TRT_PLUGIN_HELPER_HPP
