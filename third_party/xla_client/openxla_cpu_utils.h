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
// This class is designed to get accurate profile for programs.

#ifndef XLA_CLIENT_CPU_UTILS_H_
#define XLA_CLIENT_CPU_UTILS_H_

#include <chrono>
#include <memory>

#include "tsl/platform/macros.h"
#include "tsl/platform/profile_utils/i_cpu_utils_helper.h"
#include "tsl/platform/types.h"
#include "tsl/platform/profile_utils/cpu_utils.h"

namespace xla {

namespace profile_utils {
using tsl::profile_utils::CpuUtils;  // NOLINT
}  // namespace profile_utils

}  // namespace xla

#endif  // XLA_CLIENT_CPU_UTILS_H_