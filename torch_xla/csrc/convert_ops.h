#ifndef XLA_TORCH_XLA_CSRC_CONVERT_OPS_H_
#define XLA_TORCH_XLA_CSRC_CONVERT_OPS_H_

#include <c10/core/ScalarType.h>

#include <optional>

#include "torch_xla/csrc/device.h"
#include "xla/client/xla_builder.h"
#include "xla/types.h"

namespace torch_xla {

xla::XlaOp ConvertTo(xla::XlaOp op, xla::PrimitiveType from,
                     xla::PrimitiveType to);

xla::XlaOp ConvertToRaw(xla::XlaOp op, xla::PrimitiveType from,
                        xla::PrimitiveType raw_from, xla::PrimitiveType to,
                        xla::PrimitiveType raw_to);

xla::XlaOp ConvertToNumeric(xla::XlaOp op, xla::PrimitiveType from);

xla::XlaOp ConvertToNumeric(xla::XlaOp op);

// Cast the input to the given dtype. If dtype is null, no-op with the exception
// of predicates, which are converted to 8-bit unsigned integers.
xla::XlaOp CastToScalarType(xla::XlaOp input,
                            std::optional<at::ScalarType> dtype);

xla::XlaOp MaybeConvertTo(xla::XlaOp input, xla::PrimitiveType type);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_CONVERT_OPS_H_
