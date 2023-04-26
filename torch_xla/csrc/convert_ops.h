#ifndef XLA_TORCH_XLA_CSRC_CONVERT_OPS_H_
#define XLA_TORCH_XLA_CSRC_CONVERT_OPS_H_

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to,
                     const torch::lazy::BackendDevice* device);

xla::XlaOp ConvertToRaw(xla::XlaOp op, xla::PrimitiveType from,
                        xla::PrimitiveType raw_from, xla::PrimitiveType to,
                        xla::PrimitiveType raw_to,
                        const torch::lazy::BackendDevice* device);

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from);

xla::XlaOp ConvertToNumeric(xla::XlaOp op);

// Cast the input to the given dtype. If dtype is null, no-op with the exception
// of predicates, which are converted to 8-bit unsigned integers.
xla::XlaOp CastToScalarType(xla::XlaOp input,
                            c10::optional<at::ScalarType> dtype);

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_CONVERT_OPS_H_