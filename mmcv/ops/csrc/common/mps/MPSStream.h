//  Copyright © 2022 Apple Inc.

// This file is modify from:
// https://github.com/pytorch/pytorch/blob/a85d1f0bcdd02cf18d3b0517337458cb51a18cdb/aten/src/ATen/mps/MPSStream.h

#pragma once

#include <cstdint>
#include <utility>

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/util/Exception.h>
#include "MPSDevice.h"

#ifdef __OBJC__
#include <Foundation/Foundation.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
typedef id<MTLCommandQueue> MTLCommandQueue_t;
typedef id<MTLCommandBuffer> MTLCommandBuffer_t;
typedef id<MTLSharedEvent> MTLSharedEvent_t;
typedef id<MTLDevice> MTLDevice_t;
#else
typedef void* MTLCommandQueue_t;
typedef void* MTLCommandQueue;
typedef void* MTLCommandBuffer_t;
typedef void* MTLCommandBuffer;
typedef void* MTLSharedEvent_t;
typedef void* dispatch_queue_t;
typedef void* MTLDevice_t;
#define nil NULL;
#endif

namespace at {
namespace mps {

//-----------------------------------------------------------------
//  MPSStream
//-----------------------------------------------------------------

class TORCH_API MPSStream {
 public:
  enum Unchecked { UNCHECKED };
  /// Construct a MPSStream from a Stream.  This construction is checked,
  /// and will raise an error if the Stream is not, in fact, a MPS stream.
  explicit MPSStream(Stream stream);

  ~MPSStream();
  MTLCommandQueue_t commandQueue() const { return _commandQueue; };
  dispatch_queue_t queue() const { return _serialQueue; }

  MTLCommandBuffer_t commandBuffer();
  void commit(bool flush);
  void commitAndWait();
  void synchronize();

  void flush();

  /// Get the MPS device index that this stream is associated with.
  c10::DeviceIndex device_index() const { return _stream.device_index(); }

  MTLCommandQueue_t stream() const { return _commandQueue; };

  MTLDevice_t device() const { return [_commandQueue device]; }

  /// Explicit conversion to Stream.
  Stream unwrap() const { return _stream; }

 private:
  Stream _stream;
  MTLCommandQueue_t _commandQueue = nil;
  MTLCommandBuffer_t _commandBuffer = nil;
  void _flush(bool commitAndWait) const;

  dispatch_queue_t _serialQueue = nullptr;
};

/**
 * Get the current MPS stream
 */
TORCH_API MPSStream* getCurrentMPSStream();

/**
 * Get the default MPS stream
 */
TORCH_API MPSStream* getDefaultMPSStream();

//-----------------------------------------------------------------
//  MPSStreamImpl
//-----------------------------------------------------------------

class TORCH_API MPSStreamImpl {
 public:
  /**
   * Gets single instance of the MPSStream.
   */
  static MPSStream* getInstance();

 private:
  static MPSStream* _stream;
  MPSStreamImpl();
};

//-----------------------------------------------------------------
//  MPSEvent
//-----------------------------------------------------------------

struct TORCH_API MPSEvent {
  MPSEvent();
  // MPSEvent(id<MTLDevice> device);

  ~MPSEvent();
  MTLSharedEvent_t event() const { return _event; }

  void recordEvent(MPSStream* stream);
  void waitForEvent(MPSStream* queue);  // waits on the cpu
  bool queryEvent();
  uint64_t getCurrentValue() { return _currentValue; }
  void setCurrentValue(uint64_t currValue) { _currentValue = currValue; }

 private:
  bool _isRecorded = false;
  uint64_t _currentValue = 0;
  MTLSharedEvent_t _event;
};

typedef MPSEvent* mpsEvent_t;

}  // namespace mps
}  // namespace at
