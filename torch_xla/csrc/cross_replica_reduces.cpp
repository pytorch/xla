#include "torch_xla/csrc/cross_replica_reduces.h"

#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

std::vector<xla::XlaOp> BuildCrossReplicaSum(
    tensorflow::gtl::ArraySlice<const xla::XlaOp> operands, double scale,
    const std::vector<std::vector<xla::int64>>& groups) {
  std::vector<xla::ReplicaGroup> crs_groups;
  for (auto& group : groups) {
    xla::ReplicaGroup rgroup;
    for (auto replica_id : group) {
      rgroup.add_replica_ids(replica_id);
    }
    crs_groups.push_back(std::move(rgroup));
  }
  xla::XlaOp crs = xla::CrossReplicaSum(
      xla::Tuple(operands[0].builder(), operands), crs_groups);
  std::vector<xla::XlaOp> result;
  result.reserve(operands.size());
  for (size_t i = 0; i < operands.size(); ++i) {
    result.push_back(xla::GetTupleElement(crs, i));
  }
  if (scale != 1.0) {
    for (size_t i = 0; i < result.size(); ++i) {
      xla::Shape shape = XlaHelpers::ShapeOfXlaOp(result[i]);
      xla::XlaOp scaling_value = XlaHelpers::ScalarValue<float>(
          scale, shape.element_type(), result[i].builder());
      result[i] = result[i] * scaling_value;
    }
  }
  return result;
}

}  // namespace torch_xla
