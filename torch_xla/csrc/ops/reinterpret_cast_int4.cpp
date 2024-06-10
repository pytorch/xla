#include "torch_xla/csrc/ops/reinterpret_cast_int4.h"

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
    // int64_t num_in_feature = lhs_shape.dimensions()[lhs_shape.dimensions_size() - 1];
    // int64_t num_out_feature = rhs_shape.dimensions()[1] * 2;
    // xla::Shape output_shape = lhs_shape;
    // output_shape.set_dimensions(lhs_shape.dimensions_size() - 1, num_out_feature);
    // std::cout << "check dim after casting: " << output_shape << std::endl;
    // return output_shape;
}

ReinterpretCastInt4::ReinterpretCastInt4(const torch::lazy::Value& weight, 
                                         const std::vector<int>& int4_vals)
    : XlaNode(xla_reinterpret_cast_int4, {weight}, NodeOutputShape(weight),
                    /*num_outputs=*/1, torch::lazy::MHash(int4_vals)),
      int4_vals_(int4_vals) {}

torch::lazy::NodePtr ReinterpretCastInt4::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReinterpretCastInt4>(operands.at(0), int4_vals_);
}

XlaOpVector ReinterpretCastInt4::Lower(LoweringContext* loctx) const {
  xla::XlaOp weight = loctx->GetOutputOp(operand(0));
  xla::Shape weight_shape = ShapeHelper::ShapeOfXlaOp(weight);
  std::vector<xla::s4> values(int4_vals_.begin(), int4_vals_.end());
  const auto literal = xla::LiteralUtil::CreateR1(absl::Span<const xla::s4>(values));
  auto reshaped_literal = literal.Reshape(weight_shape.dimensions());
  return ReturnOp(xla::ConstantLiteral(weight.builder(), reshaped_literal.value()), loctx);
}

std::string ReinterpretCastInt4::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  return ss.str();
}

}  // namespace torch_xla
