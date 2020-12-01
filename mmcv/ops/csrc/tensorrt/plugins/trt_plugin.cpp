#include "trt_plugin.hpp"

#include "roi_align.hpp"

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
