/* Copyright 2019 The JAX Authors.

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

#ifndef TORCH_XLA_KERNEL_NANOBIND_HELPERS_H_
#define TORCH_XLA_KERNEL_NANOBIND_HELPERS_H_

#include <cstddef>
#include <stdexcept>
#include <string>

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "nanobind/nanobind.h"

namespace torch_xla {

// Packs a descriptor object into a byte string.
template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
  return std::string(absl::bit_cast<const char*>(&descriptor), sizeof(T));
}

// Unpacks a descriptor object from a byte string.
template <typename T>
absl::StatusOr<const T*> UnpackDescriptor(const char* opaque,
                                          std::size_t opaque_len) {
  if (opaque_len != sizeof(T)) {
    return absl::InternalError("Invalid size for operation descriptor.");
  }
  return absl::bit_cast<const T*>(opaque);
}

// Descriptor objects are opaque host-side objects used to pass data from JAX
// to the custom kernel launched by XLA. Currently simply treat host-side
// structures as byte-strings; this is not portable across architectures. If
// portability is needed, we could switch to using a representation such as
// protocol buffers or flatbuffers.

// Packs a descriptor object into a nanobind::bytes structure.
// UnpackDescriptor() is available in kernel_helpers.h.
template <typename T>
nanobind::bytes PackDescriptor(const T& descriptor) {
  std::string s = PackDescriptorAsString(descriptor);
  return nanobind::bytes(s.data(), s.size());
}

template <typename T>
nanobind::capsule EncapsulateFunction(T* fn) {
  return nanobind::capsule(absl::bit_cast<void*>(fn),
                           "xla._CUSTOM_CALL_TARGET");
}

}  // namespace torch_xla

#endif  // TORCH_XLA_KERNEL_NANOBIND_HELPERS_H_