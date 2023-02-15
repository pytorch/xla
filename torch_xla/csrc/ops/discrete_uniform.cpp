#include "torch_xla/csrc/ops/discrete_uniform.h"

#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

DiscreteUniform::DiscreteUniform(const torch::lazy::Value& from,
                                 const torch::lazy::Value& to,
                                 const torch::lazy::Value& seed,
                                 const xla::Shape& rng_shape)
    : XlaNode(torch::lazy::OpKind(at::aten::random), {from, to, seed},
              rng_shape,
              /*num_outputs=*/1, torch::lazy::Hash(rng_shape)) {}

torch::lazy::NodePtr DiscreteUniform::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<DiscreteUniform>(operands.at(0), operands.at(1),
                                                operands.at(2), xla_shape());
}

XlaOpVector DiscreteUniform::Lower(LoweringContext* loctx) const {
  xla::XlaOp from = loctx->GetOutputOp(operand(0));
  xla::XlaOp to = loctx->GetOutputOp(operand(1));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(2));
  return ReturnOp(RngDiscreteUniform(rng_seed, xla_shape(), from, to), loctx);
}

}  // namespace torch_xla
