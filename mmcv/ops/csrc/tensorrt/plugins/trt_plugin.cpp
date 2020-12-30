#include "trt_plugin.hpp"

#include "trt_nms.hpp"
#include "trt_roi_align.hpp"
#include "trt_scatternd.hpp"

REGISTER_TENSORRT_PLUGIN(ONNXNonMaxSuppressionDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoiAlignPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ONNXScatterNDDynamicCreator);

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
