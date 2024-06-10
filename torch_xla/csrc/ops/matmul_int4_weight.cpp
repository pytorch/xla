#include "torch_xla/csrc/ops/matmul_int4_weight.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/quant_util.h"
#include "torch_xla/csrc/shape_helper.h"
#include "xla/literal_util.h"

namespace torch_xla {

static xla::Shape NodeOutputShape(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs) {
    xla::Shape lhs_shape = GetXlaShape(lhs);
    xla::Shape rhs_shape = GetXlaShape(rhs);
    return rhs_shape;
    // int64_t num_in_feature = lhs_shape.dimensions()[lhs_shape.dimensions_size() - 1];
    // int64_t num_out_feature = rhs_shape.dimensions()[1] * 2;
    // xla::Shape output_shape = lhs_shape;
    // output_shape.set_dimensions(lhs_shape.dimensions_size() - 1, num_out_feature);
    // std::cout << "check dim after casting: " << output_shape << std::endl;
    // return output_shape;
}

ReinterpretCast4bit::ReinterpretCast4bit(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs, 
                                         const std::vector<int8_t>& int4_vals)
    : XlaNode(xla_reinterpret_cast_4ibt, {lhs, rhs}, NodeOutputShape(lhs, rhs),
                    /*num_outputs=*/1, torch::lazy::MHash(int4_vals_)),
      int4_vals_(int4_vals) {}

torch::lazy::NodePtr ReinterpretCast4bit::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReinterpretCast4bit>(operands.at(0), operands.at(1), int4_vals_);
}

XlaOpVector ReinterpretCast4bit::Lower(LoweringContext* loctx) const {
  xla::XlaOp lhs = loctx->GetOutputOp(operand(0));
  xla::Shape lhs_shape = ShapeHelper::ShapeOfXlaOp(lhs);
  xla::XlaOp rhs = loctx->GetOutputOp(operand(1));
  xla::Shape rhs_shape = ShapeHelper::ShapeOfXlaOp(rhs);

  // std::vector<int64_t> new_dims(lhs_shape.dimensions().begin(), lhs_shape.dimensions().end());
  // new_dims.insert(new_dims.begin(), 1);
  // xla::XlaOp lhs_expanded = xla::Reshape(lhs, new_dims);
                      // absl::Span<const int64_t>(new_dims.begin(), new_dims.end()));

  // xla::XlaOp weight_4bit = xla::BitcastConvertType(rhs, xla::PrimitiveType::S4);
  // xla::XlaOp weight_4bit_flattened = xla::Collapse(weight_4bit, {1, 2});
  
  // std::vector<int> values_int8 = {-1, -1, -1, -1, -1, -1, -1, -1};
  std::vector<xla::s4> values(int4_vals_.begin(), int4_vals_.end());
  // const auto literal = xla::LiteralUtil::CreateR2<xla::s4>({{xla::s4(-1), xla::s4(-1), xla::s4(-1), xla::s4(-1)}, {xla::s4(-1), xla::s4(-1), xla::s4(-1), xla::s4(-1)}});
  const auto literal = xla::LiteralUtil::CreateR1(absl::Span<const xla::s4>(values));
  auto reshaped_literal = literal.Reshape({rhs_shape.dimensions()[0], rhs_shape.dimensions()[1]});
  
  return ReturnOp(xla::ConstantLiteral(lhs.builder(), reshaped_literal.value()), loctx);
  
  // xla::DotDimensionNumbers dot_dnums;
  // // xla::Shape lhs_new_shape = ShapeHelper::ShapeOfXlaOp(lhs_expanded);
  // dot_dnums.add_lhs_contracting_dimensions(lhs_shape.dimensions_size() - 1);
  // dot_dnums.add_rhs_contracting_dimensions(0);

  // xla::XlaOp dot = xla::DotGeneral(lhs, xla::ConstantLiteral(lhs.builder(), reshaped_literal.value()), dot_dnums);
  // return ReturnOp(dot, loctx);
}

std::string ReinterpretCast4bit::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  return ss.str();
}

}  // namespace torch_xla
