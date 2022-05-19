#include "torch_xla/csrc/ops/bernoulli.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

Bernoulli::Bernoulli(const torch::lazy::Value& probability,
                     const torch::lazy::Value& seed, xla::Shape shape)
    : XlaNode(torch::lazy::OpKind(at::aten::bernoulli), {probability, seed},
              std::move(shape)) {}

torch::lazy::NodePtr Bernoulli::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Bernoulli>(operands.at(0), operands.at(1),
                                          xla_shape());
}

XlaOpVector Bernoulli::Lower(LoweringContext* loctx) const {
  xla::XlaOp probability = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(1));
  const xla::Shape& probability_shape = XlaHelpers::ShapeOfXlaOp(probability);
  xla::Shape bcast_shape(xla_shape());
  bcast_shape.set_element_type(probability_shape.element_type());
  xla::XlaOp bcast_probability = XlaHelpers::ImplicitBroadcast(
      probability, probability_shape, bcast_shape);
  return ReturnOp(
      BuildBernoulli(bcast_probability, rng_seed, xla_shape().element_type()),
      loctx);
}

}  // namespace torch_xla
