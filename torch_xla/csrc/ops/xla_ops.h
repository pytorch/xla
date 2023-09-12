#ifndef XLA_TORCH_XLA_CSRC_OPS_XLA_OPS_H_
#define XLA_TORCH_XLA_CSRC_OPS_XLA_OPS_H_

#include <mutex>
#include <string>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class OpKindWrapper {
 public:
  OpKindWrapper(const char* name) : name_(name) {}

  const torch::lazy::OpKind& operator*() const { return get(); }

  operator torch::lazy::OpKind() const { return get(); }

 private:
  const torch::lazy::OpKind& get() const {
    std::call_once(once_,
                   [this]() { op_kind_ = torch::lazy::OpKind::Get(name_); });
    return op_kind_;
  }

  const char* name_;
  mutable torch::lazy::OpKind op_kind_;
  mutable std::once_flag once_;
};

extern const OpKindWrapper xla_adam_optimizer_step;
extern const OpKindWrapper xla_all_gather;
extern const OpKindWrapper xla_all_to_all;
extern const OpKindWrapper xla_as_strided_view_update;
extern const OpKindWrapper xla_cast;
extern const OpKindWrapper xla_collective_permute;
extern const OpKindWrapper xla_cross_replica_sum;
extern const OpKindWrapper xla_device_data;
extern const OpKindWrapper xla_diagonal_view_update;
extern const OpKindWrapper xla_einsum_backward;
extern const OpKindWrapper xla_generic_slice;
extern const OpKindWrapper xla_get_dimensions_size;
extern const OpKindWrapper xla_moving_average;
extern const OpKindWrapper xla_nms;
extern const OpKindWrapper xla_not_supported;
extern const OpKindWrapper xla_optimization_barrier;
extern const OpKindWrapper xla_quantize_per_tensor;
extern const OpKindWrapper xla_recv;
extern const OpKindWrapper xla_reduce_scatter;
extern const OpKindWrapper xla_replication_pad;
extern const OpKindWrapper xla_replication_pad_backward;
extern const OpKindWrapper xla_select;
extern const OpKindWrapper xla_send;
extern const OpKindWrapper xla_sgd_optimizer_step;
extern const OpKindWrapper xla_tensor_data;
extern const OpKindWrapper xla_unselect;
extern const OpKindWrapper xla_update_slice;
extern const OpKindWrapper xla_custom_sharding;

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_XLA_OPS_H_