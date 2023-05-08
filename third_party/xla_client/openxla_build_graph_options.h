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

#ifndef XLA_CLIENT_BUILD_GRAPH_OPTIONS_H_
#define XLA_CLIENT_BUILD_GRAPH_OPTIONS_H_

#include <vector>

#include "third_party/xla_client/openxla_collective_order.h"
#include "tsl/platform/types.h"
#include "third_party/xla_client/openxla_config.pb.h"

namespace xla {

struct BuildGraphOptions {
  CallableOptions callable_options;

  // If `true`, uses Arg/Retval to implement feeds/fetches; otherwise
  // uses Recv/Send to implement feeds/fetches.
  // TODO(mrry): Remove this when the distributed runtime supports Arg/Retval.
  bool use_function_convention = false;

  static constexpr int64_t kNoCollectiveGraphKey = 0;
  int64_t collective_graph_key = kNoCollectiveGraphKey;

  // If not `kNone`, order all CollectiveReduce operations statically and
  // deterministically.  If `kEdges`, encode dependencies as explicit control
  // edges, if `kAttrs` encode as attribute on collective op.
  GraphCollectiveOrder collective_order = GraphCollectiveOrder::kNone;

  string DebugString() const;
};

}  // namespace xla

#endif  // XLA_CLIENT_BUILD_GRAPH_OPTIONS_H_