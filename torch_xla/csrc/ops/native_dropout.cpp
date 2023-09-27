#include "torch_xla/csrc/ops/native_dropout.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input) {
  xla::Shape input_shape = GetXlaShape(input);
  return xla::ShapeUtil::MakeTupleShape({input_shape, input_shape});
}

}  // namespace

NativeDropout::NativeDropout(const torch::lazy::Value& input,
                             const torch::lazy::Value& seed, float p,
                             c10::optional<bool> train)
    : XlaNode(torch::lazy::OpKind(at::aten::native_dropout), {input, seed},
              [&]() { return NodeOutputShape(input); }, 2,
              torch::lazy::MHash(p, train)),
      p_(p),
      train_(train) {}

torch::lazy::NodePtr NativeDropout::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<NativeDropout>(operands.at(0), operands.at(1),
                                              p_, train_);
}

XlaOpVector NativeDropout::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp seed = loctx->GetOutputOp(operand(1));
  return ReturnOps(BuildNativeDropout(input, seed, p_, train_), loctx);
}

}  // namespace torch_xla
