#include "torch_xla/csrc/ops/exponential.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {

Exponential::Exponential(const torch::lazy::Value& lambda,
                         const torch::lazy::Value& seed, xla::Shape shape)
    : XlaNode(torch::lazy::OpKind(at::aten::exponential), {lambda, seed},
              std::move(shape)) {}

torch::lazy::NodePtr Exponential::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Exponential>(operands.at(0), operands.at(1),
                                            xla_shape());
}

XlaOpVector Exponential::Lower(LoweringContext* loctx) const {
  xla::XlaOp lambda = loctx->GetOutputOp(operand(0));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(1));
  const xla::Shape& lambda_shape = XlaHelpers::ShapeOfXlaOp(lambda);
  xla::Shape bcast_shape(xla_shape());
  bcast_shape.set_element_type(lambda_shape.element_type());
  xla::XlaOp bcast_lambda =
      XlaHelpers::ImplicitBroadcast(lambda, lambda_shape, bcast_shape);
  return ReturnOp(
      BuildExponential(bcast_lambda, rng_seed, xla_shape().element_type()),
      loctx);
}

}  // namespace torch_xla
