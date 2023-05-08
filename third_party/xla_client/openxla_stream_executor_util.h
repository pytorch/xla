/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_CLIENT_STREAM_EXECUTOR_UTIL_H_
#define XLA_CLIENT_STREAM_EXECUTOR_UTIL_H_

#include "third_party/xla_client/openxla_tensor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "third_party/xla_client/openxla_stream_executor.h"

namespace xla {

// StreamExecutorUtil contains functions useful for interfacing
// between StreamExecutor classes and TensorFlow.
class StreamExecutorUtil {
 public:
  // Map a Tensor as a DeviceMemory object wrapping the given typed
  // buffer.
  template <typename T>
  static se::DeviceMemory<T> AsDeviceMemory(const Tensor& t) {
    T* ptr = reinterpret_cast<T*>(const_cast<char*>(t.tensor_data().data()));
    return se::DeviceMemory<T>(se::DeviceMemoryBase(ptr, t.TotalBytes()));
  }
};

}  // namespace xla

#endif  // XLA_CLIENT_STREAM_EXECUTOR_UTIL_H_