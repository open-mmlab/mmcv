#include "trt_plugin.hpp"

#include "trt_deform_conv.hpp"
#include "trt_grid_sampler.hpp"
#include "trt_nms.hpp"
#include "trt_roi_align.hpp"
#include "trt_scatternd.hpp"
#include "trt_instance_norm.hpp"

REGISTER_TENSORRT_PLUGIN(GridSamplerDynamicCreator);
REGISTER_TENSORRT_PLUGIN(DeformableConvPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(NonMaxSuppressionDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoIAlignPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ONNXScatterNDDynamicCreator);
REGISTER_TENSORRT_PLUGIN(InstanceNormalizationPluginCreator);

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
