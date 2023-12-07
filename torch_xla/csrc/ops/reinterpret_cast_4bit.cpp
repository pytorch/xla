#include "torch_xla/csrc/ops/reinterpret_cast_4bit.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/quant_util.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {

static xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
    xla::Shape s = GetXlaShape(input);
    s.add_dimensions(2);
    // s.set_element_type()
    std::cout << "check dim after casting: " << s << std::endl;
    return s;
}

ReinterpretCast4bit::ReinterpretCast4bit(const torch::lazy::Value& input, int dim)
    : XlaNode(xla_reinterpret_cast_4ibt, {input}, NodeOutputShape(input),
                    /*num_outputs=*/1, torch::lazy::MHash(dim)) {}

torch::lazy::NodePtr ReinterpretCast4bit::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<ReinterpretCast4bit>(operands.at(0));
}

XlaOpVector ReinterpretCast4bit::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::Shape input_shape = ShapeHelper::ShapeOfXlaOp(input);

  xla::XlaOp tensor_4bit = xla::BitcastConvertType(input, xla::PrimitiveType::S4);
  xla::XlaOp output = xla::ConvertElementType(input, xla::PrimitiveType::S8);
  return ReturnOp(output, loctx);
}

std::string ReinterpretCast4bit::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << "dim=" << dim_;
  return ss.str();
}

}  // namespace torch_xla
