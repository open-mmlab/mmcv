#include "trt_nms.hpp"

#include <assert.h>
#include <stdio.h>

#include <chrono>

#include "trt_serialize.hpp"

extern size_t get_onnxnms_workspace_size(size_t num_batches,
                                         size_t spatial_dimension,
                                         size_t num_classes,
                                         size_t boxes_word_size,
                                         int center_point_box);

extern void TRTONNXNMSCUDAKernelLauncher_float(
    const float *boxes, const float *scores,
    const int *max_output_boxes_per_class, const float *iou_threshold,
    const float *score_threshold, int *output, int center_point_box,
    int num_batches, int spatial_dimension, int num_classes, void *workspace,
    cudaStream_t stream);

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"NonMaxSuppression"};
}  // namespace

nvinfer1::PluginFieldCollection ONNXNonMaxSuppressionDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField>
    ONNXNonMaxSuppressionDynamicCreator::mPluginAttributes;

ONNXNonMaxSuppressionDynamic::ONNXNonMaxSuppressionDynamic(
    const std::string &name, int centerPointBox)
    : mLayerName(name), mCenterPointBox(centerPointBox), mNumberInputs(0) {}

ONNXNonMaxSuppressionDynamic::ONNXNonMaxSuppressionDynamic(
    const std::string name, const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mCenterPointBox);
}

nvinfer1::IPluginV2DynamicExt *ONNXNonMaxSuppressionDynamic::clone() const {
  ONNXNonMaxSuppressionDynamic *plugin =
      new ONNXNonMaxSuppressionDynamic(mLayerName, mCenterPointBox);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs ONNXNonMaxSuppressionDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 2;
  auto num_batches = inputs[0].d[0];
  auto spatial_dimension = inputs[0].d[1];
  auto num_classes = inputs[1].d[1];
  ret.d[0] = exprBuilder.operation(
      nvinfer1::DimensionOperation::kPROD, *num_batches,
      *exprBuilder.operation(nvinfer1::DimensionOperation::kPROD,
                             *spatial_dimension, *num_classes));
  ret.d[1] = exprBuilder.constant(3);

  return ret;
}

bool ONNXNonMaxSuppressionDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  if (pos < nbInputs) {
    switch (pos) {
      case 0:
        // boxes
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      case 1:
        // scores
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      case 2:
        // max_output_boxes_per_class
        return inOut[pos].type == nvinfer1::DataType::kINT32 &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      case 3:
        // iou_threshold
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      case 4:
        // score_threshold
        return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      default:
        return true;
    }
  } else {
    switch (pos - nbInputs) {
      case 0:
        // selected_indices
        return inOut[pos].type == nvinfer1::DataType::kINT32 &&
               inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
      default:
        return true;
    }
  }
  return true;
}

void ONNXNonMaxSuppressionDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {
  mNumberInputs = nbInputs;
}

size_t ONNXNonMaxSuppressionDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  size_t boxes_word_size = mmcv::getElementSize(inputs[0].type);
  size_t num_batches = inputs[0].dims.d[0];
  size_t spatial_dimension = inputs[0].dims.d[1];
  size_t num_classes = inputs[1].dims.d[1];

  return get_onnxnms_workspace_size(num_batches, spatial_dimension, num_classes,
                                    boxes_word_size, mCenterPointBox);
}

int ONNXNonMaxSuppressionDynamic::enqueue(
    const nvinfer1::PluginTensorDesc *inputDesc,
    const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs,
    void *const *outputs, void *workSpace, cudaStream_t stream) {
  int num_batches = inputDesc[0].dims.d[0];
  int spatial_dimension = inputDesc[0].dims.d[1];
  int num_classes = inputDesc[1].dims.d[1];

  const float *boxes = (const float *)inputs[0];
  const float *scores = (const float *)inputs[1];
  const int *max_output_boxes_per_class =
      (mNumberInputs >= 3) ? (const int *)inputs[2] : nullptr;
  const float *iou_threshold =
      (mNumberInputs >= 4) ? (const float *)inputs[3] : nullptr;
  const float *score_threshold =
      (mNumberInputs >= 5) ? (const float *)inputs[4] : nullptr;

  int *output = (int *)outputs[0];

  TRTONNXNMSCUDAKernelLauncher_float(
      boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold,
      output, mCenterPointBox, num_batches, spatial_dimension, num_classes,
      workSpace, stream);

  return 0;
}

nvinfer1::DataType ONNXNonMaxSuppressionDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return nvinfer1::DataType::kINT32;
}

// IPluginV2 Methods
const char *ONNXNonMaxSuppressionDynamic::getPluginType() const {
  return PLUGIN_NAME;
}

const char *ONNXNonMaxSuppressionDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int ONNXNonMaxSuppressionDynamic::getNbOutputs() const { return 1; }

int ONNXNonMaxSuppressionDynamic::initialize() { return 0; }

void ONNXNonMaxSuppressionDynamic::terminate() {}

size_t ONNXNonMaxSuppressionDynamic::getSerializationSize() const {
  return sizeof(mCenterPointBox);
}

void ONNXNonMaxSuppressionDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mCenterPointBox);
}

void ONNXNonMaxSuppressionDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void ONNXNonMaxSuppressionDynamic::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ONNXNonMaxSuppressionDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

ONNXNonMaxSuppressionDynamicCreator::ONNXNonMaxSuppressionDynamicCreator() {
  mPluginAttributes.clear();
  // mPluginAttributes.emplace_back(nvinfer1::PluginField("center_point_box"));
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *ONNXNonMaxSuppressionDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *ONNXNonMaxSuppressionDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
ONNXNonMaxSuppressionDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *ONNXNonMaxSuppressionDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  int centerPointBox = 0;
  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("center_point_box") == 0) {
      centerPointBox = static_cast<const int *>(fc->fields[i].data)[0];
    }
  }

  ONNXNonMaxSuppressionDynamic *plugin =
      new ONNXNonMaxSuppressionDynamic(name, centerPointBox);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *ONNXNonMaxSuppressionDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  auto plugin =
      new ONNXNonMaxSuppressionDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void ONNXNonMaxSuppressionDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *ONNXNonMaxSuppressionDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}
