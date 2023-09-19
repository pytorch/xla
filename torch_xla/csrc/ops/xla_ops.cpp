#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {

const OpKindWrapper xla_adam_optimizer_step("xla::adam_optimizer_step");
const OpKindWrapper xla_all_gather("xla::all_gather");
const OpKindWrapper xla_all_to_all("xla::all_to_all");
const OpKindWrapper xla_as_strided_view_update("xla::as_strided_view_update");
const OpKindWrapper xla_cast("xla::cast");
const OpKindWrapper xla_collective_permute("xla::collective_permute");
const OpKindWrapper xla_cross_replica_sum("xla::cross_replica_sum");
const OpKindWrapper xla_device_data("xla::device_data");
const OpKindWrapper xla_dequantize_per_tensor("xla::dequantize_per_tensor");
const OpKindWrapper xla_diagonal_view_update("xla::diagonal_view_update");
const OpKindWrapper xla_einsum_backward("xla::einsum_backward");
const OpKindWrapper xla_generic_slice("xla::generic_slice");
const OpKindWrapper xla_get_dimensions_size("xla::xla_get_dimensions_size");
const OpKindWrapper xla_moving_average("xla::moving_average");
const OpKindWrapper xla_nms("xla::nms");
const OpKindWrapper xla_not_supported("xla::not_supported");
const OpKindWrapper xla_optimization_barrier("xla::optimization_barrier");
const OpKindWrapper xla_quantize_per_tensor("xla::quantize_per_tensor");
const OpKindWrapper xla_recv("xla::recv");
const OpKindWrapper xla_reduce_scatter("xla::reduce_scatter");
const OpKindWrapper xla_replication_pad("xla::replication_pad");
const OpKindWrapper xla_replication_pad_backward(
    "xla::replication_pad_backward");
const OpKindWrapper xla_select("xla::select");
const OpKindWrapper xla_send("xla::send");
const OpKindWrapper xla_sgd_optimizer_step("xla::sgd_optimizer_step");
const OpKindWrapper xla_tensor_data("xla::tensor_data");
const OpKindWrapper xla_unselect("xla::unselect");
const OpKindWrapper xla_update_slice("xla::update_slice");
const OpKindWrapper xla_custom_sharding("xla::custom_sharding");

}  // namespace torch_xla
