#include "trt_plugin.hpp"

#include "trt_roi_align.hpp"

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
