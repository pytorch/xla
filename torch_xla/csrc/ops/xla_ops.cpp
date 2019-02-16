#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

const OpKindWrapper xla_device_data("xla::device_data");
const OpKindWrapper xla_cross_replica_sum("xla::cross_replica_sum");
const OpKindWrapper xla_cast("xla::cast");

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
