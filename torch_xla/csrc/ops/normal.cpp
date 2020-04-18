#include "torch_xla/csrc/ops/normal.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"

namespace torch_xla {
namespace ir {
namespace ops {

Normal::Normal(const Value& mean, const Value& std, const Value& seed)
    : Node(ir::OpKind(at::aten::normal), {mean, std, seed}, mean.shape()) {}

NodePtr Normal::Clone(OpList operands) const {
  return MakeNode<Normal>(operands.at(0), operands.at(1), operands.at(2));
}

XlaOpVector Normal::Lower(LoweringContext* loctx) const {
  xla::XlaOp mean = loctx->GetOutputOp(operand(0));
  xla::XlaOp std = loctx->GetOutputOp(operand(1));
  xla::XlaOp rng_seed = loctx->GetOutputOp(operand(2));
  return ReturnOp(
      RngNormal(rng_seed, XlaHelpers::ShapeOfXlaOp(mean), mean, std), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
