#include "lazy_xla/csrc/compiler/token_handler.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "lazy_xla/csrc/compiler/convert_ops.h"
#include "lazy_xla/csrc/compiler/helpers.h"

namespace torch_xla {
namespace {

xla::XlaOp SliceOneToken(xla::XlaOp input) {
  const xla::Shape& input_shape = compiler::XlaHelpers::ShapeOfXlaOp(input);
  xla::int64 input_rank = input_shape.rank();
  if (input_rank > 0) {
    xla::GatherDimensionNumbers dim_numbers;
    for (xla::int64 i = 0; i < input_rank; ++i) {
      dim_numbers.add_collapsed_slice_dims(i);
      dim_numbers.add_start_index_map(i);
    }
    dim_numbers.set_index_vector_dim(0);

    std::vector<xla::int64> slice_sizes(input_rank, 1);
    xla::XlaOp indices = xla::Zeros(
        input.builder(),
        xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32, {input_rank}));
    input = xla::Gather(input, indices, dim_numbers, slice_sizes);
  }
  return input;
}

}  // namespace

xla::XlaOp TokenHandler::GetInput(xla::XlaOp input,
                                  const xla::Shape* input_shape) {
  if (input_shape == nullptr) {
    input_shape = &compiler::XlaHelpers::ShapeOfXlaOp(input);
  }
  // Token is always a numeric zero, so adding to input does not change input.
  return input + MaybeConvertTo(token_, input_shape->element_type());
}

xla::XlaOp TokenHandler::GetNewToken(xla::XlaOp result) {
  xla::XlaOp slice = SliceOneToken(result);
  // Token is always a numeric zero, and multiplying it for one element of the
  // result will still leave it as zero.
  token_ =
      token_ * MaybeConvertTo(slice, compiler::XlaHelpers::TypeOfXlaOp(token_));
  return token_;
}

}  // namespace torch_xla
