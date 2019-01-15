#include "ops/cross_replica_sum.h"

#include "lowering_context.h"
#include "ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

CrossReplicaSum::CrossReplicaSum(const NodeOperand& operand,
                                 std::vector<std::vector<xla::int64>> groups)
    : Node(xla_cross_replica_sum, {operand}, operand.node->shape()),
      groups_(std::move(groups)) {}

XlaOpVector Generic::Lower(LoweringContext* loctx) const {
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
