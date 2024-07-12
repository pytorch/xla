#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {

const OpKindWrapper xla_adam_optimizer_step("xla::adam_optimizer_step");
const OpKindWrapper xla_all_gather("xla::all_gather");
const OpKindWrapper xla_all_to_all("xla::all_to_all");
const OpKindWrapper xla_as_strided_view_update("xla::as_strided_view_update");
const OpKindWrapper xla_cast("xla::cast");
const OpKindWrapper xla_collective_permute("xla::collective_permute");
const OpKindWrapper xla_cross_replica_sum("xla::cross_replica_sum");
const OpKindWrapper xla_custom_call("xla::custom_call");
const OpKindWrapper xla_device_data("xla::device_data");
const OpKindWrapper xla_dequantize_tensor("xla::dequantize_tensor");
const OpKindWrapper xla_diagonal_view_update("xla::diagonal_view_update");
const OpKindWrapper xla_dynamic_expand("xla::dynamic_expand");
const OpKindWrapper xla_dynamic_view("xla::dynamic_view");
const OpKindWrapper xla_einsum_backward("xla::einsum_backward");
const OpKindWrapper xla_generic_slice("xla::generic_slice");
const OpKindWrapper xla_get_dimensions_size("xla::xla_get_dimensions_size");
const OpKindWrapper xla_mark_tensor("xla::mark_tensor");
const OpKindWrapper xla_moving_average("xla::moving_average");
const OpKindWrapper xla_nms("xla::nms");
const OpKindWrapper xla_not_supported("xla::not_supported");
const OpKindWrapper xla_optimization_barrier("xla::optimization_barrier");
const OpKindWrapper xla_quantize_tensor("xla::quantize_tensor");
const OpKindWrapper xla_recv("xla::recv");
const OpKindWrapper xla_reduce_scatter("xla::reduce_scatter");
const OpKindWrapper xla_cast_int4("xla::cast_int4");
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
const OpKindWrapper xla_tpu_custom_call("xla::tpu_custom_call");
const OpKindWrapper xla_gpu_custom_call("xla::gpu_custom_call");

}  // namespace torch_xla
