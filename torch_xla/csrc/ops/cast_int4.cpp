#include "torch_xla/csrc/ops/cast_int4.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/quant_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/literal_util.h"

namespace torch_xla {

static xla::Shape NodeOutputShape(const torch::lazy::Value& weight) {
  xla::Shape out_shape = GetXlaShape(weight);
  out_shape.set_element_type(xla::PrimitiveType::S4);
  return out_shape;
}

CastInt4::CastInt4(const torch::lazy::Value& weight,
                   const std::vector<int>& int4_vals)
    : XlaNode(xla_cast_int4, {weight}, NodeOutputShape(weight),
              /*num_outputs=*/1, torch::lazy::MHash(int4_vals)),
      int4_vals_(int4_vals) {}

torch::lazy::NodePtr CastInt4::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<CastInt4>(operands.at(0), int4_vals_);
}

XlaOpVector CastInt4::Lower(LoweringContext* loctx) const {
  xla::XlaOp weight = loctx->GetOutputOp(operand(0));
  xla::Shape weight_shape = ShapeHelper::ShapeOfXlaOp(weight);
  std::vector<xla::s4> values(int4_vals_.begin(), int4_vals_.end());
  const auto literal =
      xla::LiteralUtil::CreateR1(absl::Span<const xla::s4>(values));
  auto reshaped_literal = literal.Reshape(weight_shape.dimensions());
  return ReturnOp(
      xla::ConstantLiteral(weight.builder(), reshaped_literal.value()), loctx);
}

std::string CastInt4::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  return ss.str();
}

}  // namespace torch_xla
