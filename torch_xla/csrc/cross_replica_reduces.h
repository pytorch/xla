#pragma once

#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

enum class AllReduceType {
  kSum,
  kMin,
  kMax,
  kMul,
  kOr,
  kAnd,
};

std::vector<xla::XlaOp> BuildAllReduce(
    AllReduceType reduce_type,
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands, xla::XlaOp token,
    double scale, const std::vector<std::vector<xla::int64>>& groups);

}  // namespace torch_xla
