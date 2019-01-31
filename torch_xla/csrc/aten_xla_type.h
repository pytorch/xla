#pragma once

#include "aten_xla_type_base.h"

namespace torch_xla {

// Base ATEN Type class where the XLA specific overrides should be defined.
class AtenXlaType : public AtenXlaTypeBase {
 public:
  AtenXlaType(at::TensorTypeId type_id, bool is_variable, bool is_undefined);
};

}  // namespace torch_xla
