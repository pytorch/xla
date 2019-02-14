#include "torch_xla/csrc/ops/cross_replica_sum.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

CrossReplicaSum::CrossReplicaSum(const Value& operand,
                                 std::vector<std::vector<xla::int64>> groups)
    : Node(xla_cross_replica_sum, {operand}, operand.shape(),
           /*num_outputs=*/1, xla::util::MHash(groups)),
      groups_(std::move(groups)) {}

std::string CrossReplicaSum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << " groups=(";
  for (size_t i = 0; i < groups_.size(); ++i) {
    const auto& group = groups_[i];
    ss << (i == 0 ? "(" : ",(");
    for (size_t j = 0; j < group.size(); ++j) {
      if (j > 0) {
        ss << ",";
      }
      ss << group[j];
    }
    ss << ")";
  }
  ss << ")";
  return ss.str();
}

XlaOpVector CrossReplicaSum::Lower(LoweringContext* loctx) const {
  std::vector<xla::ReplicaGroup> crs_groups;
  for (auto& group : groups_) {
    xla::ReplicaGroup rgroup;
    for (auto replica_id : group) {
      rgroup.add_replica_ids(replica_id);
    }
    crs_groups.push_back(std::move(rgroup));
  }
  xla::XlaOp op = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::CrossReplicaSum(op, crs_groups), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
