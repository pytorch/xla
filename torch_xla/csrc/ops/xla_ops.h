#pragma once

#include <mutex>
#include <string>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class OpKindWrapper {
 public:
  OpKindWrapper(const char* name) : name_(name) {}

  const OpKind& operator*() const { return get(); }

  operator OpKind() const { return get(); }

 private:
  const OpKind& get() const {
    std::call_once(once_, [this]() { op_kind_ = OpKind::Get(name_); });
    return op_kind_;
  }

  const char* name_;
  mutable OpKind op_kind_;
  mutable std::once_flag once_;
};

extern const OpKindWrapper xla_as_strided_view_update;
extern const OpKindWrapper xla_cast;
extern const OpKindWrapper xla_cross_replica_sum;
extern const OpKindWrapper xla_device_data;
extern const OpKindWrapper xla_diagonal_view_update;
extern const OpKindWrapper xla_generic_slice;
extern const OpKindWrapper xla_get_dimensions_size;
extern const OpKindWrapper xla_moving_average;
extern const OpKindWrapper xla_not_supported;
extern const OpKindWrapper xla_select;
extern const OpKindWrapper xla_tensor_data;
extern const OpKindWrapper xla_token;
extern const OpKindWrapper xla_unselect;
extern const OpKindWrapper xla_update_slice;

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
