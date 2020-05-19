#include "torch_xla/csrc/token_handler.h"

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

xla::XlaOp SliceOneToken(xla::XlaOp input) {
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  for (xla::int64 dim = 0; dim < input_shape.rank(); ++dim) {
    input = xla::SliceInDim(input, 0, 1, 1, dim);
  }
  return input;
}

}  // namespace

xla::XlaOp TokenHandler::GetInput(xla::XlaOp input,
                                  const xla::Shape* input_shape) {
  if (input_shape == nullptr) {
    input_shape = &XlaHelpers::ShapeOfXlaOp(input);
  }
  // Token is always a numeric zero, so adding to input does not change input.
  return input + MaybeConvertTo(token_, input_shape->element_type());
}

xla::XlaOp TokenHandler::GetNewToken(xla::XlaOp result) {
  xla::XlaOp slice = SliceOneToken(result);
  // Token is always a numeric zero, and multiplying it for one element of the
  // result will still leave it as zero.
  token_ = token_ * MaybeConvertTo(slice, XlaHelpers::TypeOfXlaOp(token_));
  return token_;
}

}  // namespace torch_xla
