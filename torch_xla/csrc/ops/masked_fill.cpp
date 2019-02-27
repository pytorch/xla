#include "torch_xla/csrc/ops/masked_fill.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/scalar.h"

namespace torch_xla {
namespace ir {
namespace ops {

MaskedFill::MaskedFill(const Value& input, const Value& mask, at::Scalar value)
    : Node(OpKind(at::aten::masked_fill), {input, mask}, input.shape(),
           /*num_outputs=*/1, ScalarHash(value)),
      value_(std::move(value)) {}

XlaOpVector MaskedFill::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp mask = loctx->GetOutputOp(operand(1));
  xla::XlaOp zero = XlaHelpers::ScalarValue(
      0, XlaHelpers::ShapeOfXlaOp(mask).element_type(), loctx->builder());
  xla::XlaOp mask_pred = xla::Ne(mask, zero);
  // Input shape is the same as output shape.
  const xla::Shape& input_shape = shape();
  xla::XlaOp value =
      xla::Broadcast(XlaHelpers::ScalarValue(value_, input_shape.element_type(),
                                             input.builder()),
                     input_shape.dimensions());
  return ReturnOp(xla::Select(mask_pred, value, input), loctx);
}

std::string MaskedFill::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", value=" << value_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
