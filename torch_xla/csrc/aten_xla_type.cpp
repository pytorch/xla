#include "aten_xla_type.h"

namespace torch_xla {

AtenXlaType::AtenXlaType(at::TensorTypeId type_id, bool is_variable,
                         bool is_undefined)
    : AtenXlaTypeBase(type_id, is_variable, is_undefined) {}

}  // namespace torch_xla
