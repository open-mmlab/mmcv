#include "roi_align.hpp"

#include <assert.h>

#include <chrono>

#include "serialize.hpp"

namespace {
static const char *PLUGIN_VERSION{"1"};
static const char *PLUGIN_NAME{"MMCVRoiAlign"};
}  // namespace

nvinfer1::PluginFieldCollection RoiAlignPluginDynamicCreator::mFC{};
std::vector<nvinfer1::PluginField>
    RoiAlignPluginDynamicCreator::mPluginAttributes(
        {nvinfer1::PluginField("out_height"),
         nvinfer1::PluginField("out_width"),
         nvinfer1::PluginField("spatial_scale"),
         nvinfer1::PluginField("sampling_ratio"),
         nvinfer1::PluginField("pool_mode"), nvinfer1::PluginField("aligned")});

RoiAlignPluginDynamic::RoiAlignPluginDynamic(const std::string &name,
                                             int outWidth, int outHeight,
                                             float spatialScale,
                                             int sampleRatio, int poolMode,
                                             bool aligned)
    : mLayerName(name),
      mOutWidth(outWidth),
      mOutHeight(outHeight),
      mSpatialScale(spatialScale),
      mSampleRatio(sampleRatio),
      mPoolMode(poolMode),
      mAligned(aligned) {}

RoiAlignPluginDynamic::RoiAlignPluginDynamic(const std::string name,
                                             const void *data, size_t length)
    : mLayerName(name) {
  deserialize_value(&data, &length, &mOutWidth);
  deserialize_value(&data, &length, &mOutHeight);
  deserialize_value(&data, &length, &mSpatialScale);
  deserialize_value(&data, &length, &mSampleRatio);
  deserialize_value(&data, &length, &mPoolMode);
  deserialize_value(&data, &length, &mAligned);
}

nvinfer1::IPluginV2DynamicExt *RoiAlignPluginDynamic::clone() const {
  RoiAlignPluginDynamic *plugin = new RoiAlignPluginDynamic(
      mLayerName, mOutWidth, mOutHeight, mSpatialScale, mSampleRatio, mPoolMode,
      mAligned);
  plugin->setPluginNamespace(getPluginNamespace());

  return plugin;
}

nvinfer1::DimsExprs RoiAlignPluginDynamic::getOutputDimensions(
    int outputIndex, const nvinfer1::DimsExprs *inputs, int nbInputs,
    nvinfer1::IExprBuilder &exprBuilder) {
  nvinfer1::DimsExprs ret;
  ret.nbDims = 4;
  ret.d[0] = inputs[1].d[0];
  ret.d[1] = inputs[0].d[1];
  ret.d[2] = exprBuilder.constant(mOutHeight);
  ret.d[3] = exprBuilder.constant(mOutWidth);

  return ret;
}

bool RoiAlignPluginDynamic::supportsFormatCombination(
    int pos, const nvinfer1::PluginTensorDesc *inOut, int nbInputs,
    int nbOutputs) {
  // const auto *in = inOut;
  // const auto *out = inOut + nbInputs;
  return inOut[pos].type == nvinfer1::DataType::kFLOAT &&
         inOut[pos].format == nvinfer1::TensorFormat::kLINEAR;
}

void RoiAlignPluginDynamic::configurePlugin(
    const nvinfer1::DynamicPluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::DynamicPluginTensorDesc *outputs, int nbOutputs) {}

size_t RoiAlignPluginDynamic::getWorkspaceSize(
    const nvinfer1::PluginTensorDesc *inputs, int nbInputs,
    const nvinfer1::PluginTensorDesc *outputs, int nbOutputs) const {
  return 0;
}

int RoiAlignPluginDynamic::enqueue(const nvinfer1::PluginTensorDesc *inputDesc,
                                   const nvinfer1::PluginTensorDesc *outputDesc,
                                   const void *const *inputs,
                                   void *const *outputs, void *workSpace,
                                   cudaStream_t stream) {
  //   int num_rois = inputDesc[0].dims.d[0];
  //   int batch_size = inputDesc[1].dims.d[0];
  //   int channels = inputDesc[1].dims.d[1];

  //   const int kMaxFeatMap = 10;
  //   int heights[kMaxFeatMap];
  //   int widths[kMaxFeatMap];
  //   float strides[kMaxFeatMap];

  //   int num_feats = mFeatmapStrides.size();
  //   for (int i = 0; i < num_feats; ++i) {
  //     heights[i] = inputDesc[i + 1].dims.d[2];
  //     widths[i] = inputDesc[i + 1].dims.d[3];
  //     strides[i] = mFeatmapStrides[i];
  //   }

  //   const void *rois = inputs[0];
  //   const void *const *feats = inputs + 1;

  //   roi_extractor<float>((float *)outputs[0], (const float *)rois, num_rois,
  //                        feats, num_feats, batch_size, channels, &heights[0],
  //                        &widths[0], &strides[0], mOutSize, mSampleNum,
  //                        mRoiScaleFactor, mFinestScale, mAligned, stream);

  return 0;
}

nvinfer1::DataType RoiAlignPluginDynamic::getOutputDataType(
    int index, const nvinfer1::DataType *inputTypes, int nbInputs) const {
  return inputTypes[0];
}

// IPluginV2 Methods
const char *RoiAlignPluginDynamic::getPluginType() const { return PLUGIN_NAME; }

const char *RoiAlignPluginDynamic::getPluginVersion() const {
  return PLUGIN_VERSION;
}

int RoiAlignPluginDynamic::getNbOutputs() const { return 1; }

int RoiAlignPluginDynamic::initialize() { return 0; }

void RoiAlignPluginDynamic::terminate() {}

size_t RoiAlignPluginDynamic::getSerializationSize() const {
  return sizeof(mOutWidth) + sizeof(mOutHeight) + sizeof(mSpatialScale) +
         sizeof(mSampleRatio) + sizeof(mPoolMode) + sizeof(mAligned);
}

void RoiAlignPluginDynamic::serialize(void *buffer) const {
  serialize_value(&buffer, mOutWidth);
  serialize_value(&buffer, mOutHeight);
  serialize_value(&buffer, mSpatialScale);
  serialize_value(&buffer, mSampleRatio);
  serialize_value(&buffer, mPoolMode);
  serialize_value(&buffer, mAligned);
}

void RoiAlignPluginDynamic::destroy() {
  // This gets called when the network containing plugin is destroyed
  delete this;
}

void RoiAlignPluginDynamic::setPluginNamespace(const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *RoiAlignPluginDynamic::getPluginNamespace() const {
  return mNamespace.c_str();
}

////////////////////// creator /////////////////////////////

RoiAlignPluginDynamicCreator::RoiAlignPluginDynamicCreator() {
  mFC.nbFields = mPluginAttributes.size();
  mFC.fields = mPluginAttributes.data();
}

const char *RoiAlignPluginDynamicCreator::getPluginName() const {
  return PLUGIN_NAME;
}

const char *RoiAlignPluginDynamicCreator::getPluginVersion() const {
  return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection *
RoiAlignPluginDynamicCreator::getFieldNames() {
  return &mFC;
}

nvinfer1::IPluginV2 *RoiAlignPluginDynamicCreator::createPlugin(
    const char *name, const nvinfer1::PluginFieldCollection *fc) {
  int outWidth = 7;
  int outHeight = 7;
  float spatialScale = 1.0;
  int sampleRatio = 0;
  int poolMode = -1;
  bool aligned = true;

  for (int i = 0; i < fc->nbFields; i++) {
    if (fc->fields[i].data == nullptr) {
      continue;
    }
    std::string field_name(fc->fields[i].name);

    if (field_name.compare("out_height") == 0) {
      outHeight = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("out_width") == 0) {
      outWidth = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("spatial_scale") == 0) {
      spatialScale = static_cast<const float *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("sample_ratio") == 0) {
      sampleRatio = static_cast<const int *>(fc->fields[i].data)[0];
    }

    if (field_name.compare("pool_mode") == 0) {
      int data_size = fc->fields[i].length;
      const char *data_start = static_cast<const char *>(fc->fields[i].data);
      std::string poolModeStr(data_start, data_size);
      if (poolModeStr == "avg") {
        poolMode = 0;
      } else if (poolModeStr == "max") {
        poolMode = 1;
      } else {
        std::cout << "Unknown pool mode \"" << poolModeStr << "\"."
                  << std::endl;
      }
      assert(poolMode >= 0);
    }

    if (field_name.compare("aligned") == 0) {
      int aligned_int = static_cast<const int *>(fc->fields[i].data)[0];
      aligned = aligned_int != 0;
    }
  }

  assert(outHeight > 0);
  assert(outWidth > 0);
  assert(spatialScale > 0.);
  assert(poolMode >= 0);

  RoiAlignPluginDynamic *plugin = new RoiAlignPluginDynamic(
      name, outWidth, outHeight, spatialScale, sampleRatio, poolMode, aligned);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

nvinfer1::IPluginV2 *RoiAlignPluginDynamicCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength) {
  auto plugin = new RoiAlignPluginDynamic(name, serialData, serialLength);
  plugin->setPluginNamespace(getPluginNamespace());
  return plugin;
}

void RoiAlignPluginDynamicCreator::setPluginNamespace(
    const char *libNamespace) {
  mNamespace = libNamespace;
}

const char *RoiAlignPluginDynamicCreator::getPluginNamespace() const {
  return mNamespace.c_str();
}