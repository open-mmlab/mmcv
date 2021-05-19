// Modified from:
// https://github.com/NVIDIA/TensorRT/blob/master/plugin/instanceNormalizationPlugin/instanceNormalizationPlugin.cpp

#include "trt_instance_norm.hpp"

#include <cuda_fp16.h>
#include <stdexcept>

#include "trt_serialize.hpp"

using namespace nvinfer1;

cudnnStatus_t convert_trt2cudnn_dtype(nvinfer1::DataType trt_dtype,
                                      cudnnDataType_t* cudnn_dtype) {
  switch (trt_dtype) {
    case nvinfer1::DataType::kFLOAT:
      *cudnn_dtype = CUDNN_DATA_FLOAT;
      break;
    case nvinfer1::DataType::kHALF:
      *cudnn_dtype = CUDNN_DATA_HALF;
      break;
    default:
      return CUDNN_STATUS_BAD_PARAM;
  }
  return CUDNN_STATUS_SUCCESS;
}

namespace {
constexpr const char* PLUGIN_VERSION{"1"};
constexpr const char* PLUGIN_NAME{"MMCVInstanceNormalization"};
}  // namespace

PluginFieldCollection InstanceNormalizationPluginCreator::mFC{};
std::vector<PluginField> InstanceNormalizationPluginCreator::mPluginAttributes;

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    const std::string& name, float epsilon)
    : mLayerName(name), mEpsilon(epsilon) {}

InstanceNormalizationPlugin::InstanceNormalizationPlugin(
    const std::string& name, void const* serialData, size_t serialLength)
    : mLayerName(name) {
  deserialize_value(&serialData, &serialLength, &mEpsilon);
}

InstanceNormalizationPlugin::~InstanceNormalizationPlugin() {}

// InstanceNormalizationPlugin returns one output.
int InstanceNormalizationPlugin::getNbOutputs() const { return 1; }

DimsExprs InstanceNormalizationPlugin::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs* inputs, int nbInputs,
    nvinfer1::IExprBuilder& exprBuilder) {
  nvinfer1::DimsExprs output(inputs[0]);
  return output;
}

int InstanceNormalizationPlugin::initialize() { return 0; }

void InstanceNormalizationPlugin::terminate() {}

size_t InstanceNormalizationPlugin::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  int n = inputs[0].dims.d[0];
  int c = inputs[0].dims.d[1];
  int elem_size = mmcv::getElementSize(inputs[1].type);
  return mmcv::getAlignedSize(n * c * elem_size) * 2;
}

int InstanceNormalizationPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc,
    const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) {
  nvinfer1::Dims input_dims = inputDesc[0].dims;
  int n = input_dims.d[0];
  int c = input_dims.d[1];
  int h = input_dims.d[2];
  int w = input_dims.nbDims > 3 ? input_dims.d[3] : 1;
  int elem_size = mmcv::getElementSize(inputDesc[1].type);

  void* n_scales = (void*)workspace;
  void* n_bias = (void*)(workspace + mmcv::getAlignedSize(n * c * elem_size));

  const void* scales = (const void*)inputs[1];
  const void* bias = (const void*)inputs[2];

  for (int i = 0; i < n; ++i) {
    cudaMemcpyAsync(n_scales + i * c * elem_size, scales, c * elem_size,
                    cudaMemcpyDeviceToDevice, stream);
    cudaMemcpyAsync(n_bias + i * c * elem_size, bias, c * elem_size,
                    cudaMemcpyDeviceToDevice, stream);
  }

  cudnnSetTensor4dDescriptor(_b_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1,
                             n * c, 1, 1);
  cudnnDataType_t cudnn_dtype{};
  convert_trt2cudnn_dtype(inputDesc[0].type, &cudnn_dtype);
  cudnnSetTensor4dDescriptor(_x_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c,
                             h, w);
  cudnnSetTensor4dDescriptor(_y_desc, CUDNN_TENSOR_NCHW, cudnn_dtype, 1, n * c,
                             h, w);
  float alpha = 1;
  float beta = 0;
  void const* x_ptr = inputs[0];
  void* y_ptr = outputs[0];
  cudnnSetStream(_cudnn_handle, stream);
  // Note: Use of CUDNN_BATCHNORM_SPATIAL_PERSISTENT can cause numerical
  //       overflows (NaNs) for fp32 data in some circumstances. The lower-
  //       performance CUDNN_BATCHNORM_SPATIAL should be used if this is not
  //       acceptable.
  cudnnBatchNormalizationForwardTraining(
      _cudnn_handle, CUDNN_BATCHNORM_SPATIAL_PERSISTENT, &alpha, &beta, _x_desc,
      x_ptr, _y_desc, y_ptr, _b_desc, n_scales, n_bias, 1., nullptr, nullptr,
      mEpsilon, nullptr, nullptr);
  return 0;
}

size_t InstanceNormalizationPlugin::getSerializationSize() const {
  return serialized_size(mEpsilon);
}

void InstanceNormalizationPlugin::serialize(void* buffer) const {
  serialize_value(&buffer, mEpsilon);
}

bool InstanceNormalizationPlugin::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs,
    int nbOutputs) {
  return ((inOut[pos].type == nvinfer1::DataType::kFLOAT ||
           inOut[pos].type == nvinfer1::DataType::kHALF) &&
          inOut[pos].format == nvinfer1::PluginFormat::kNCHW &&
          inOut[pos].type == inOut[0].type);
}

const char* InstanceNormalizationPlugin::getPluginType() const {
  return PLUGIN_NAME;
}

const char* InstanceNormalizationPlugin::getPluginVersion() const {
  return PLUGIN_VERSION;
}

void InstanceNormalizationPlugin::destroy() { delete this; }

IPluginV2DynamicExt* InstanceNormalizationPlugin::clone() const {
  auto* plugin = new InstanceNormalizationPlugin{mLayerName, mEpsilon};
  plugin->setPluginNamespace(mPluginNamespace.c_str());
  return plugin;
}

// Set plugin namespace
void InstanceNormalizationPlugin::setPluginNamespace(
    const char* pluginNamespace) {
  mPluginNamespace = pluginNamespace;
}

const char* InstanceNormalizationPlugin::getPluginNamespace() const {
  return mPluginNamespace.c_str();
}

nvinfer1::DataType InstanceNormalizationPlugin::getOutputDataType(
    int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// Attach the plugin object to an execution context and grant the plugin the
// access to some context resource.
void InstanceNormalizationPlugin::attachToContext(cudnnContext* cudnnContext,
                                                  cublasContext* cublasContext,
                                                  IGpuAllocator* gpuAllocator) {
  _cudnn_handle = cudnnContext;
  cudnnCreateTensorDescriptor(&_b_desc);
  cudnnCreateTensorDescriptor(&_x_desc);
  cudnnCreateTensorDescriptor(&_y_desc);
}

// Detach the plugin object from its execution context.
void InstanceNormalizationPlugin::detachFromContext() {
  cudnnDestroyTensorDescriptor(_y_desc);
  cudnnDestroyTensorDescriptor(_x_desc);
  cudnnDestroyTensorDescriptor(_b_desc);
}

void InstanceNormalizationPlugin::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {}

// InstanceNormalizationPluginCreator methods
InstanceNormalizationPluginCreator::InstanceNormalizationPluginCreator() {
  mPluginAttributes.clear();
  mPluginAttributes.emplace_back(
      PluginField("epsilon", nullptr, PluginFieldType::kFLOAT32, 1));

  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char* InstanceNormalizationPluginCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char* InstanceNormalizationPluginCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const PluginFieldCollection*
InstanceNormalizationPluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc) {
  float epsilon = 1e-5;
  const PluginField* fields = fc->fields;
  for (int i = 0; i < fc->nbFields; ++i) {
    const char* attrName = fields[i].name;
    if (!strcmp(attrName, "epsilon")) {
      epsilon = *(static_cast<const float*>(fields[i].data));
    }
  }

  InstanceNormalizationPlugin* obj =
      new InstanceNormalizationPlugin(name, epsilon);
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

IPluginV2DynamicExt* InstanceNormalizationPluginCreator::deserializePlugin(
    const char* name, const void* serialData, size_t serialLength) {
  InstanceNormalizationPlugin* obj =
      new InstanceNormalizationPlugin{name, serialData, serialLength};
  obj->setPluginNamespace(mNamespace.c_str());
  return obj;
}

void InstanceNormalizationPluginCreator::setPluginNamespace(
    const char* libNamespace) {
  mNamespace = libNamespace;
}

const char* InstanceNormalizationPluginCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
