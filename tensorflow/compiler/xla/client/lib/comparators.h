#pragma once

#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

inline XlaComputation CreateScalarLtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

inline XlaComputation CreateScalarGtComputation(
    const std::vector<PrimitiveType>& operand_types, XlaBuilder* builder) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
