#include "torch_xla/csrc/ops/native_dropout.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
    xla::Shape input_shape = GetXlaShape(input);
    return xla::ShapeUtil::MakeTupleShape(
                        {input_shape, input_shape});
}

}  // namespace

NativeDropout::NativeDropout(const torch::lazy::Value& input, 
                        const float p,
                        const c10::optional<bool> train,
                        const torch::lazy::Value& seed)
    : XlaNode(torch::lazy::OpKind(at::aten::native_dropout), {input, p, train, seed},
              NodeOutputShape(input),
              /*num_outputs=*/2) {}

torch::lazy::NodePtr NativeDropout::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<NativeDropout>(operands.at(0), operands.at(1),
                                            operands.at(2), operants.at(3));
}

XlaOpVector NativeDropout::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp p = loctx->GetOutputOp(operand(1));
  xla::XlaOp train = loctx->GetOutputOp(operand(2));
  xla::XlaOp seed = loctx->GetOutputOp(operand(3));
  return ReturnOp(BuildNativeDropout(input, p, seed, train),
                  loctx);
}

}  // namespace torch_xla
