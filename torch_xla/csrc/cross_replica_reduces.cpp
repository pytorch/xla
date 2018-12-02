#include "cross_replica_reduces.h"
#include <vector>
#include "helpers.h"

namespace torch {
namespace jit {

xla::XlaOp BuildCrossReplicaSum(const xla::XlaOp& operand, int num_replicas) {
  xla::XlaOp crs = xla::CrossReplicaSum(operand);
  auto shape = XlaHelpers::ShapeOfXlaOp(operand);
  auto scaling_value = XlaHelpers::ScalarValue<float>(
      1.0 / num_replicas, shape.element_type(), operand.builder());
  return crs * xla::Broadcast(scaling_value, XlaHelpers::ShapeSizes(shape));
}

}  // namespace jit
}  // namespace torch
