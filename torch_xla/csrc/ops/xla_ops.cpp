#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

const OpKindWrapper xla_as_strided_view_update("xla::as_strided_view_update");
const OpKindWrapper xla_cast("xla::cast");
const OpKindWrapper xla_cross_replica_sum("xla::cross_replica_sum");
const OpKindWrapper xla_device_data("xla::device_data");
const OpKindWrapper xla_diagonal_view_update("xla::diagonal_view_update");
const OpKindWrapper xla_generic_slice("xla::generic_slice");
const OpKindWrapper xla_get_dimensions_size("xla::xla_get_dimensions_size");
const OpKindWrapper xla_moving_average("xla::moving_average");
const OpKindWrapper xla_not_supported("xla::not_supported");
const OpKindWrapper xla_select("xla::select");
const OpKindWrapper xla_tensor_data("xla::tensor_data");
const OpKindWrapper xla_token("xla::token");
const OpKindWrapper xla_unselect("xla::unselect");
const OpKindWrapper xla_update_slice("xla::update_slice");

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
