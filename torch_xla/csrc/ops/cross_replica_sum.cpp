#include "torch_xla/csrc/ops/cross_replica_sum.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(tensorflow::gtl::ArraySlice<const Value> operands) {
  std::vector<xla::Shape> tuple_shapes;
  tuple_shapes.reserve(operands.size());
  for (auto& operand : operands) {
    tuple_shapes.push_back(operand.shape());
  }
  return xla::ShapeUtil::MakeTupleShape(tuple_shapes);
}

}  // namespace

CrossReplicaSum::CrossReplicaSum(
    tensorflow::gtl::ArraySlice<const Value> operands, double scale,
    std::vector<std::vector<xla::int64>> groups)
    : Node(xla_cross_replica_sum, operands,
           [&]() { return NodeOutputShape(operands); },
           /*num_outputs=*/operands.size(), xla::util::MHash(scale, groups)),
      scale_(scale),
      groups_(std::move(groups)) {}

NodePtr CrossReplicaSum::Clone(OpList operands) const {
  return MakeNode<CrossReplicaSum>(operands, scale_, groups_);
}

XlaOpVector CrossReplicaSum::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  inputs.reserve(operands().size());
  for (auto& input : operands()) {
    inputs.push_back(loctx->GetOutputOp(input));
  }
  return ReturnOps(BuildCrossReplicaSum(inputs, scale_, groups_), loctx);
}

std::string CrossReplicaSum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", scale=" << scale_ << ", groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    ss << (i == 0 ? "(" : ",(");
    ss << absl::StrJoin(groups_[i], ", ") << ")";
  }
  ss << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
