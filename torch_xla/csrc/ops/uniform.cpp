#include "torch_xla/csrc/ops/uniform.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "tensorflow/compiler/xla/xla_client/xla_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"

namespace torch_xla {
namespace ir {
namespace ops {

Uniform::Uniform(const Value& from, const Value& to, const Value& seed,
                 const xla::Shape& rng_shape)
    : Node(ir::OpKind(at::aten::uniform), {from, to, seed}, rng_shape,
           /*num_outputs=*/1, xla::util::ShapeHash(rng_shape)) {}

NodePtr Uniform::Clone(OpList operands) const {
  return MakeNode<Uniform>(operands.at(0), operands.at(1), operands.at(2),
                           shape());
}

XlaOpVector Uniform::Lower(LoweringContext* loctx) const {
  xla::XlaOp from = loctx->GetOutputOp(operand(0));
  xla::XlaOp to = loctx->GetOutputOp(operand(1));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(2));
  return ReturnOp(RngUniform(rng_seed, shape(), from, to), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
