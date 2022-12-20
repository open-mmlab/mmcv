#ifndef PARROTS_DEVICE_REGISTRY_H
#define PARROTS_DEVICE_REGISTRY_H

#define MAX_DEVICE_TYPES 8  // currently x86, cuda, camb, hip, ascend

#include <parrots/extension.hpp>
#include <part/devices.hpp>

namespace parrots {

// declare Registry
template <typename F, F f>
class DeviceRegistry;

// Template specialization
template <typename Ret, typename... Args, Ret (*f)(Args...)>
class DeviceRegistry<Ret (*)(Args...), f> {
 public:
  using FunctionType = Ret (*)(Args...);

  void registerArch(const Arch& device, FunctionType function) {
    funcs_[size_t(device)] = function;
  }

  FunctionType find(const Arch& device) const { return funcs_[size_t(device)]; }

  static DeviceRegistry& instance() {
    static DeviceRegistry instance;
    return instance;
  }

 private:
  DeviceRegistry() {
    for (size_t i = 0; i < MAX_DEVICE_TYPES; i++) {
      funcs_[i] = nullptr;
    }
  };
  FunctionType funcs_[MAX_DEVICE_TYPES];
};

// dispatch
template <typename R, typename... Args>
auto Dispatch(const R& registry, string_t name, Context& ctx, Args&&... args) {
  auto arch = ctx.getProxy().arch();
  auto f_ptr = registry.find(arch);
  PARROTS_CHECKARGS(f_ptr != nullptr)
      << name + ": implementation for arch " + archName(arch) + " not found.\n";
  return f_ptr(ctx, std::forward<Args>(args)...);
}

#define DEVICE_REGISTRY(key) DeviceRegistry<decltype(&(key)), key>::instance()

#define REGISTER_DEVICE_IMPL(key, device, arch, value) \
  struct key##_##device##_registerer {                 \
    key##_##device##_registerer() {                    \
      DEVICE_REGISTRY(key).registerArch(arch, value);  \
    }                                                  \
  };                                                   \
  static key##_##device##_registerer _##key##_##device##_registerer;

#define DISPATCH_DEVICE_IMPL(key, ...) \
  Dispatch(DEVICE_REGISTRY(key), #key, __VA_ARGS__)

#endif  // PARROTS_DEVICE_REGISTRY

}  // namespace parrots
