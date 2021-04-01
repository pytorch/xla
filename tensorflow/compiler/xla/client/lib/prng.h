#pragma once

#include <functional>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape.h"

namespace xla {

struct RngOutput {
  XlaOp value;
  XlaOp state;
};

using BitGeneratorTy = std::function<RngOutput(XlaOp key, XlaOp initial_state,
                                               const Shape& shape)>;

RngOutput ThreeFryBitGenerator(XlaOp key, XlaOp initial_state,
                               const xla::Shape& shape) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

RngOutput PhiloxBitGenerator(XlaOp key, XlaOp initial_state,
                             const Shape& shape) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}
std::pair<XlaOp, XlaOp> ScramblePhiloxKey(XlaOp key) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

RngOutput UniformFloatingPointDistribution(XlaOp key, XlaOp initial_state,
                                           BitGeneratorTy bit_generator,
                                           XlaOp minval, XlaOp maxval,
                                           const xla::Shape& shape) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

RngOutput UniformIntDistribution(XlaOp key, XlaOp initial_state,
                                 BitGeneratorTy bit_generator, XlaOp minval,
                                 XlaOp maxval, const xla::Shape& shape) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

RngOutput NormalFloatingPointDistribution(XlaOp key, XlaOp initial_state,
                                          BitGeneratorTy bit_generator,
                                          const xla::Shape& shape) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

XlaOp ConcatScalars(XlaBuilder* builder, absl::Span<const XlaOp> scalars) {
  LTC_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
