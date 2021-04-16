#pragma once

#include "lazy_xla/csrc/compiler/xla_lowering_context.h"

namespace torch_lazy_tensors {
namespace compiler {

XlaOpVector BuildAmpForeachNonFiniteCheckAndUnscale(
    const XlaOpVector& inputs, const xla::XlaOp& found_inf_float,
    const xla::XlaOp& inv_scale);

}
}  // namespace torch_lazy_tensors
