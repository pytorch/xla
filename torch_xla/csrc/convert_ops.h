#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/types.h"
#include "torch_xla/csrc/device.h"

namespace torch_xla {

xla::XlaOp ConvertTo(const xla::XlaOp& op, xla::PrimitiveType from,
                     xla::PrimitiveType to, const Device* device);

}  // namespace torch_xla
