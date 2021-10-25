#ifndef PYTORCH_DEVICE_REGISTRY_H
#define PYTORCH_DEVICE_REGISTRY_H

#include <torch/extension.h>

#include <cassert>
#include <functional>
#include <map>
#include <type_traits>

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
    const at::Device& device0, int index) {
  return {index, device0};
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
std::pair<int, at::Device> CheckDeviceConsistency(const at::Device& device0,
                                                  int index, T&& t,
                                                  Args&&... args) {
  auto device1 = std::forward<T>(t).device();
  if (device1.type() != device0.type() || device1.index() != device0.index()) {
    return {index, device1};
  }
  return CheckDeviceConsistency(device0, index + 1,
                                std::forward<Args>(args)...);
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
  if (inconsist.first < sizeof...(Args)) {
    fprintf(stderr, "%s: at param %d, inconsistent device: %s vs %s\n", name,
            inconsist.first, inconsist.second.str().c_str(),
            device.str().c_str());
    std::abort();
  }
  auto f_ptr = registry.Find(device.type());
  if (!f_ptr) {
    fprintf(stderr, "%s: implementation for device %s not found\n", name,
            device.str().c_str());
    std::abort();
  }
  return f_ptr(std::forward<Args>(args)...);
}

// helper macro

#define REGISTRY(key) DeviceRegistry<decltype(&(key)), key>::instance()

#define REGISTER(key, device, value)                \
  struct key##_##device##_registerer {              \
    key##_##device##_registerer() {                 \
      REGISTRY(key).Register(at::k##device, value); \
    }                                               \
  };                                                \
  static key##_##device##_registerer _##key##_##device##_registerer;

#define DISPATCH(key, ...) Dispatch(REGISTRY(key), #key, __VA_ARGS__)

#endif  // PYTORCH_DEVICE_REGISTRY
