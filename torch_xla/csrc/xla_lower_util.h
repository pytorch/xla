#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

xla::XlaOp CreateMatMul(const xla::XlaOp& lhs, const xla::XlaOp& rhs);

xla::XlaOp BuildDropout(const xla::XlaOp& input, float probability);

std::vector<xla::XlaOp> CreateBroadcastTensors(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands);

}  // namespace torch_xla
