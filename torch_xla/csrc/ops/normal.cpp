#include "torch_xla/csrc/ops/normal.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& mean, const Value& std,
                           xla::uint64 seed) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp rng_seed = XlaHelpers::ScalarValue(seed, xla::PrimitiveType::U64,
                                                  operands[0].builder());
    return BuildNormal(operands[0], operands[1], rng_seed);
  };
  return InferOutputShape({mean.shape(), std.shape()}, lower_for_shape_fn);
}

}  // namespace

Normal::Normal(const Value& mean, const Value& std, xla::uint64 seed)
    : Node(ir::OpKind(at::aten::normal), {mean, std},
           [&]() { return NodeOutputShape(mean, std, seed); },
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
  return ReturnOp(BuildNormal(mean, std, rng_seed), loctx);
}

std::string Normal::ToString() const {
  std::stringstream ss;
  ss << Node::ToString();
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
