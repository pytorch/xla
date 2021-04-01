#pragma once

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

typedef std::function<StatusOr<XlaOp>(absl::Span<const XlaOp>, XlaBuilder*)>
    WhileLoopHelperConditionFunction;

typedef std::function<StatusOr<std::vector<XlaOp>>(absl::Span<const XlaOp>,
                                                   XlaBuilder*)>
    WhileLoopHelperBodyFunction;

inline StatusOr<std::vector<XlaOp>> WhileLoopHelper(
    const WhileLoopHelperConditionFunction& condition_function,
    const WhileLoopHelperBodyFunction& body_function,
    absl::Span<const XlaOp> initial_values, absl::string_view name,
    XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla