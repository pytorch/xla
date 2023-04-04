#include "torch_xla/csrc/ops/multinomial.h"

#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input, int64_t num_samples, bool replacement) {
  auto input_shape = GetXlaShape(input);
  int64_t rank = input_shape.rank();
  XLA_CHECK(rank == 1 || rank == 2);
  if (!replacement)
    XLA_CHECK_LE(num_samples, input_shape.dimensions(1));
  if (rank == 2) {
    return  xla::ShapeUtil::MakeShape(xla::PrimitiveType::S64, {input_shape.dimensions(0), num_samples});
  } 
  return xla::ShapeUtil::MakeShape(xla::PrimitiveType::S64, {num_samples});
}

}  // namespace

Multinomial::Multinomial(const torch::lazy::Value& input, const torch::lazy::Value& seed,
                         int64_t num_samples, bool replacement)
    : XlaNode(torch::lazy::OpKind(at::aten::multinomial), {input, seed},
    [&]() {
                return NodeOutputShape(input, num_samples, replacement);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(num_samples, replacement)),
    num_samples_(num_samples),
    replacement_(replacement) {}

torch::lazy::NodePtr Multinomial::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Multinomial>(operands.at(0), operands.at(1),
                                            num_samples_, replacement_);
}

XlaOpVector Multinomial::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp seed = loctx->GetOutputOp(operand(1));
  return ReturnOp(BuildMultinomial(input, num_samples_, replacement_, seed), loctx);
}

}  // namespace torch_xla
