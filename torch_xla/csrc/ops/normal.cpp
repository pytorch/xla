#include "torch_xla/csrc/ops/normal.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"

namespace torch_xla {

Normal::Normal(const torch::lazy::Value& mean, const torch::lazy::Value& std,
               const torch::lazy::Value& seed)
    : XlaNode(torch::lazy::OpKind(at::aten::normal), {mean, std, seed},
              GetXlaShape(mean)) {}

torch::lazy::NodePtr Normal::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Normal>(operands.at(0), operands.at(1),
                                       operands.at(2));
}

XlaOpVector Normal::Lower(LoweringContext* loctx) const {
  xla::XlaOp mean = loctx->GetOutputOp(operand(0));
  xla::XlaOp std = loctx->GetOutputOp(operand(1));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(2));
  return ReturnOp(
      RngNormal(rng_seed, XlaHelpers::ShapeOfXlaOp(mean), mean, std), loctx);
}

}  // namespace torch_xla
