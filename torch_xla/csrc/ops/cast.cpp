#include "torch_xla/csrc/ops/cast.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, xla::PrimitiveType type) {
  xla::Shape shape = input.shape();
  shape.set_element_type(type);
  return shape;
}

}  // namespace

Cast::Cast(const Value& input, xla::PrimitiveType type)
    : Node(xla_cast, {input}, NodeOutputShape(input, type),
           /*num_outputs=*/1, xla::util::MHash(static_cast<int>(type))),
      type_(type) {}

Cast::Cast(const Value& input, at::ScalarType dtype,
           c10::optional<at::ScalarType> stype)
    : Node(xla_cast, {input},
           NodeOutputShape(input,
                           MakeXlaPrimitiveType(dtype, /*device=*/nullptr)),
           /*num_outputs=*/1,
           xla::util::MHash(101, static_cast<int>(dtype),
                            OptionalOr<int>(stype, -1))),
      type_(MakeXlaPrimitiveType(dtype, /*device=*/nullptr)),
      dtype_(dtype),
      stype_(stype) {}

NodePtr Cast::Clone(OpList operands) const {
  return dtype_ ? MakeNode<Cast>(operands.at(0), *dtype_, stype_)
                : MakeNode<Cast>(operands.at(0), type_);
}

XlaOpVector Cast::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const xla::Shape& input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::PrimitiveType raw_from =
      stype_ ? TensorTypeToRawXlaType(*stype_) : input_shape.element_type();
  xla::PrimitiveType raw_to = dtype_ ? TensorTypeToRawXlaType(*dtype_) : type_;
  xla::XlaOp output =
      ConvertToRaw(input, input_shape.element_type(), raw_from, type_, raw_to,
                   /*device=*/nullptr);
  return ReturnOp(output, loctx);
}

std::string Cast::ToString() const {
  std::stringstream ss;
  ss << Node::ToString()
     << ", type=" << xla::primitive_util::LowercasePrimitiveTypeName(type_);
  if (dtype_) {
    ss << ", dtype=" << *dtype_;
  }
  if (stype_) {
    ss << ", stype=" << *stype_;
  }
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
