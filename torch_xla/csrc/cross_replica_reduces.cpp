#include "torch_xla/csrc/cross_replica_reduces.h"

#include <vector>

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::XlaOp BuildCrossReplicaSum(
    const xla::XlaOp& operand, double scale,
    const std::vector<std::vector<xla::int64>>& groups) {
  std::vector<xla::ReplicaGroup> crs_groups;
  for (auto& group : groups) {
    xla::ReplicaGroup rgroup;
    for (auto replica_id : group) {
      rgroup.add_replica_ids(replica_id);
    }
    crs_groups.push_back(std::move(rgroup));
  }
  xla::XlaOp crs = xla::CrossReplicaSum(operand, crs_groups);
  if (scale != 1.0) {
    xla::Shape shape = XlaHelpers::ShapeOfXlaOp(operand);
    xla::XlaOp scaling_value = XlaHelpers::ScalarValue<float>(
        scale, shape.element_type(), operand.builder());
    crs = crs * scaling_value;
  }
  return crs;
}

}  // namespace torch_xla
