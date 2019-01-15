#include "ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

const OpKindWrapper xla_device_data("xla::device_data");
const OpKindWrapper xla_add("xla::add");
const OpKindWrapper xla_sub("xla::sub");
const OpKindWrapper xla_mul("xla::mul");
const OpKindWrapper xla_div("xla::div");

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
