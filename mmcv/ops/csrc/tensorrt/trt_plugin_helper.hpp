#pragma once

#include "NvInferPlugin.h"

namespace mmcv {
template <typename T>
class PluginRegistrarWithNamespace {
 public:
  PluginRegistrarWithNamespace(const char* nspace) {
    getPluginRegistry()->registerCreator(instance, nspace);
  }

 private:
  T instance{};
};

#define REGISTER_TENSORRT_PLUGIN_WITH_NSPACE(name, nspace) \
  static mmcv::PluginRegistrarWithNamespace<name>          \
      PluginRegistrarWithNamespace##name(#nspace);

}  // namespace mmcv
