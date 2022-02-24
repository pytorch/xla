#include "torch_xla/csrc/ops/uniform.h"

#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch/csrc/lazy/core/ir.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Uniform::Uniform(const Value& from, const Value& to, const Value& seed,
                 const xla::Shape& rng_shape)
    : Node(torch::lazy::OpKind(at::aten::uniform), {from, to, seed}, rng_shape,
           /*num_outputs=*/1, torch::lazy::Hash(rng_shape)) {}

NodePtr Uniform::Clone(OpList operands) const {
  return ir::MakeNode<Uniform>(operands.at(0), operands.at(1), operands.at(2),
                               shape());
}

XlaOpVector Uniform::Lower(LoweringContext* loctx) const {
  xla::XlaOp from = loctx->GetOutputOp(operand_with_shape(0));
  xla::XlaOp to = loctx->GetOutputOp(operand_with_shape(1));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand_with_shape(2));
  return ReturnOp(RngUniform(rng_seed, shape(), from, to), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
