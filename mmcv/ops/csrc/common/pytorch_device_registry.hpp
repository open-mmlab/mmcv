#ifndef PYTORCH_DEVICE_REGISTRY_H
#define PYTORCH_DEVICE_REGISTRY_H

#include <torch/extension.h>

#include <cassert>
#include <functional>
#include <map>
#include <type_traits>

inline std::string GetDeviceStr(const at::Device& device) {
  std::string str = DeviceTypeName(device.type(), true);
  if (device.has_index()) {
    str.push_back(':');
    str.append(std::to_string(device.index()));
  }
  return str;
}

// Registry
template <typename F, F f>
class DeviceRegistry;

template <typename Ret, typename... Args, Ret (*f)(Args...)>
class DeviceRegistry<Ret (*)(Args...), f> {
 public:
  using FunctionType = Ret (*)(Args...);

  void Register(at::DeviceType device, FunctionType function) {
    entries_.insert({device, function});
  }

  FunctionType Find(at::DeviceType device) const {
    auto it = entries_.find(device);
    if (it != entries_.end()) {
      return it->second;
    }
    return nullptr;
  }

  static DeviceRegistry& instance() {
    static DeviceRegistry inst;
    return inst;
  }

 private:
  DeviceRegistry() = default;
  std::map<at::DeviceType, FunctionType> entries_;
};

// get device of first tensor param

template <typename T, typename... Args,
          std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value,
                           bool> = true>
at::Device GetFirstTensorDevice(T&& t, Args&&... args) {
  return std::forward<T>(t).device();
}
template <typename T, typename... Args,
          std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value,
                           bool> = true>
at::Device GetFirstTensorDevice(T&& t, Args&&... args) {
  return GetFirstTensorDevice(std::forward<Args>(args)...);
}

// check device consistency

inline std::pair<int, at::Device> CheckDeviceConsistency(
    const at::Device& device, int index) {
  return {index, device};
}

template <typename T, typename... Args,
          std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value,
                           bool> = true>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
                                                  int index, T&& t,
                                                  Args&&... args);

template <typename T, typename... Args,
          std::enable_if_t<std::is_same<std::decay_t<T>, at::Tensor>::value,
                           bool> = true>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
                                                  int index, T&& t,
                                                  Args&&... args) {
  auto new_device = std::forward<T>(t).device();
  if (new_device.type() != device.type() ||
      new_device.index() != device.index()) {
    return {index, new_device};
  }
  return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

template <
    typename T, typename... Args,
    std::enable_if_t<!std::is_same<std::decay_t<T>, at::Tensor>::value, bool>>
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device,
                                                  int index, T&& t,
                                                  Args&&... args) {
  return CheckDeviceConsistency(device, index + 1, std::forward<Args>(args)...);
}

// dispatch

template <typename R, typename... Args>
auto Dispatch(const R& registry, const char* name, Args&&... args) {
  auto device = GetFirstTensorDevice(std::forward<Args>(args)...);
  auto inconsist =
      CheckDeviceConsistency(device, 0, std::forward<Args>(args)...);
  TORCH_CHECK(inconsist.first >= int(sizeof...(Args)), name, ": at param ",
              inconsist.first,
              ", inconsistent device: ", GetDeviceStr(inconsist.second).c_str(),
              " vs ", GetDeviceStr(device).c_str(), "\n")
  auto f_ptr = registry.Find(device.type());
  TORCH_CHECK(f_ptr != nullptr, name, ": implementation for device ",
              GetDeviceStr(device).c_str(), " not found.\n")
  return f_ptr(std::forward<Args>(args)...);
}

// helper macro

#define DEVICE_REGISTRY(key) DeviceRegistry<decltype(&(key)), key>::instance()

#define REGISTER_DEVICE_IMPL(key, device, value)           \
  struct key##_##device##_registerer {                     \
    key##_##device##_registerer() {                        \
      DEVICE_REGISTRY(key).Register(at::k##device, value); \
    }                                                      \
  };                                                       \
  static key##_##device##_registerer _##key##_##device##_registerer;

#define DISPATCH_DEVICE_IMPL(key, ...) \
  Dispatch(DEVICE_REGISTRY(key), #key, __VA_ARGS__)

#endif  // PYTORCH_DEVICE_REGISTRY
