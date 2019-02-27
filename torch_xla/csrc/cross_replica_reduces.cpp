#include "torch_xla/csrc/cross_replica_reduces.h"
#include <vector>
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {

xla::XlaOp BuildCrossReplicaSum(const xla::XlaOp& operand, int num_replicas) {
  xla::XlaOp crs = xla::CrossReplicaSum(operand);
  auto shape = XlaHelpers::ShapeOfXlaOp(operand);
  auto scaling_value = XlaHelpers::ScalarValue<float>(
      1.0 / num_replicas, shape.element_type(), operand.builder());
  return crs * xla::Broadcast(scaling_value, shape.dimensions());
}

}  // namespace torch_xla
