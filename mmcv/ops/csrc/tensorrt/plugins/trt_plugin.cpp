#include "trt_plugin.hpp"

#include "trt_cummaxmin.hpp"
#include "trt_deform_conv.hpp"
#include "trt_grid_sampler.hpp"
#include "trt_instance_norm.hpp"
#include "trt_modulated_deform_conv.hpp"
#include "trt_nms.hpp"
#include "trt_roi_align.hpp"
#include "trt_scatternd.hpp"

REGISTER_TENSORRT_PLUGIN(CumMaxPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(CumMinPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(GridSamplerDynamicCreator);
REGISTER_TENSORRT_PLUGIN(DeformableConvPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ModulatedDeformableConvPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(NonMaxSuppressionDynamicCreator);
REGISTER_TENSORRT_PLUGIN(RoIAlignPluginDynamicCreator);
REGISTER_TENSORRT_PLUGIN(ONNXScatterNDDynamicCreator);
REGISTER_TENSORRT_PLUGIN(InstanceNormalizationDynamicCreator);

extern "C" {
bool initLibMMCVInferPlugins() { return true; }
}  // extern "C"
