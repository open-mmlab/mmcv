#include "trt_plugin.hpp"

#include "trt_roi_align.hpp"

REGISTER_TENSORRT_PLUGIN(RoiAlignPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ONNXScatterNDDynamicCreator);

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
