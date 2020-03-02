#include "torch_xla/csrc/ops/normal.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"

namespace torch_xla {
namespace ir {
namespace ops {

Normal::Normal(const Value& mean, const Value& std, xla::uint64 seed)
    : Node(ir::OpKind(at::aten::normal), {mean, std}, mean.shape(),
           /*num_outputs=*/1, xla::util::MHash(seed)),
      seed_(seed) {}

NodePtr Normal::Clone(OpList operands) const {
  return MakeNode<Normal>(operands.at(0), operands.at(1), seed_);
}

XlaOpVector Normal::Lower(LoweringContext* loctx) const {
  xla::XlaOp mean = loctx->GetOutputOp(operand(0));
  xla::XlaOp std = loctx->GetOutputOp(operand(1));
  xla::XlaOp rng_seed =
      XlaHelpers::ScalarValue(seed_, xla::PrimitiveType::U64, mean.builder());
  return ReturnOp(
      RngNormal(rng_seed, XlaHelpers::ShapeOfXlaOp(mean), mean, std), loctx);
}

std::string Normal::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", seed=" << seed_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
