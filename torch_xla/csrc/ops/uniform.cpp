#include "torch_xla/csrc/ops/uniform.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/random.h"

namespace torch_xla {
namespace ir {
namespace ops {

Uniform::Uniform(const Value& from, const Value& to,
                 const xla::Shape& rng_shape, xla::uint64 seed)
    : Node(ir::OpKind(at::aten::uniform), {from, to}, rng_shape,
           /*num_outputs=*/1, xla::util::MHash(seed)),
      seed_(seed) {}

NodePtr Uniform::Clone(OpList operands) const {
  return MakeNode<Uniform>(operands.at(0), operands.at(1), shape(), seed_);
}

XlaOpVector Uniform::Lower(LoweringContext* loctx) const {
  xla::XlaOp from = loctx->GetOutputOp(operand(0));
  xla::XlaOp to = loctx->GetOutputOp(operand(1));
  xla::XlaOp rng_seed =
      XlaHelpers::ScalarValue(seed_, xla::PrimitiveType::U64, from.builder());
  return ReturnOp(RngUniform(rng_seed, shape(), from, to), loctx);
}

std::string Uniform::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", seed=" << seed_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
