/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

#ifndef XLA_CLIENT_MEMORY_TYPES_H_
#define XLA_CLIENT_MEMORY_TYPES_H_

#include "third_party/xla_client/openxla_op.h"
#include "third_party/xla_client/openxla_types.h"

namespace xla {

class NodeDef;

// Returns into *{input,output}_memory_types the memory type of each
// {input,output} tensor.
//
// REQUIRES: * '*_memory_types' is not nullptr.
//           * def has all attrs specified (e.g. using AddDefaultsToNodeDef()).
Status MemoryTypesForNode(const OpRegistryInterface* op_registry,
                          const DeviceType& device_type, const NodeDef& ndef,
                          MemoryTypeVector* input_memory_types,
                          MemoryTypeVector* output_memory_types);

}  // namespace xla

#endif  // XLA_CLIENT_MEMORY_TYPES_H_