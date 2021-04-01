#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

inline XlaComputation CreateScalarAddComputation(PrimitiveType type,
                                                 XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaComputation CreateScalarGeComputation(PrimitiveType type,
                                                XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaComputation CreateScalarMaxComputation(PrimitiveType type,
                                                 XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaComputation CreateScalarMinComputation(PrimitiveType type,
                                                 XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaComputation CreateScalarAndComputation(PrimitiveType type,
                                                 XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaComputation CreateScalarOrComputation(PrimitiveType type,
                                                XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaComputation CreateScalarIdentityWithZeroComputation(
    PrimitiveType type, XlaBuilder* builder) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp ArgMaxTwoPass(XlaOp input, PrimitiveType output_type, int axis,
                           bool tie_low = true) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp ArgMinTwoPass(XlaOp input, PrimitiveType output_type, int axis,
                           bool tie_low = true) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
