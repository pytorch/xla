#include "torch_xla/csrc/ops/matmul_int4_weight.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/quant_util.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {

static xla::Shape NodeOutputShape(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs) {
    xla::Shape lhs_shape = GetXlaShape(lhs);
    xla::Shape rhs_shape = GetXlaShape(rhs);
    int64_t bs = lhs_shape.dimensions()[0];
    int64_t num_in_feature = lhs_shape.dimensions()[1];
    int64_t num_out_feature = rhs_shape.dimensions()[1] * 2;
    xla::Shape output_shape = lhs_shape;
    output_shape.set_dimensions(1, num_out_feature);
    std::cout << "check dim after casting: " << output_shape << std::endl;
    return output_shape;
}

ReinterpretCast4bit::ReinterpretCast4bit(const torch::lazy::Value& lhs, const torch::lazy::Value& rhs)
    : XlaNode(xla_reinterpret_cast_4ibt, {lhs, rhs}, NodeOutputShape(lhs, rhs),
                    /*num_outputs=*/1, torch::lazy::MHash()) {}

torch::lazy::NodePtr ReinterpretCast4bit::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReinterpretCast4bit>(operands.at(0), operands.at(1));
}

XlaOpVector ReinterpretCast4bit::Lower(LoweringContext* loctx) const {
  xla::XlaOp lhs = loctx->GetOutputOp(operand(0));
  xla::Shape lhs_shape = ShapeHelper::ShapeOfXlaOp(lhs);
  xla::XlaOp rhs = loctx->GetOutputOp(operand(1));
  xla::Shape rhs_shape = ShapeHelper::ShapeOfXlaOp(rhs);

  xla::XlaOp weight_4bit = xla::BitcastConvertType(rhs, xla::PrimitiveType::S4);
  xla::XlaOp weight_4bit_flattened = xla::Collapse(weight_4bit, {1, 2});
  xla::XlaOp dot = xla::Dot(lhs, weight_4bit_flattened);
  return ReturnOp(dot, loctx);
}

std::string ReinterpretCast4bit::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString();
  return ss.str();
}

}  // namespace torch_xla
