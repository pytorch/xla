#include "torch_xla/csrc/ops/cross_replica_sum.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

CrossReplicaSum::CrossReplicaSum(const Value& operand, double scale,
                                 std::vector<std::vector<xla::int64>> groups)
    : Node(xla_cross_replica_sum, {operand}, operand.shape(),
           /*num_outputs=*/1, xla::util::MHash(scale, groups)),
      scale_(scale),
      groups_(std::move(groups)) {}

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

XlaOpVector CrossReplicaSum::Lower(LoweringContext* loctx) const {
  xla::XlaOp op = loctx->GetOutputOp(operand(0));
  xla::XlaOp crs = BuildCrossReplicaSum(op, scale_, groups_);
  return ReturnOp(crs, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
