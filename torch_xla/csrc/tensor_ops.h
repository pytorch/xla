#pragma once

#include "torch_xla/csrc/tensor.h"

// Certain tensor operations can be expressed in terms of other tensor
// operations. Add their implementations here instead of the main XLATensor
// class.

namespace torch_xla {
namespace tensor_ops {

XLATensor Cross(const XLATensor& input, const XLATensor& other, xla::int64 dim);

XLATensor SmoothL1Loss(const XLATensor& input, const XLATensor& target,
                       xla::int64 reduction);

}  // namespace tensor_ops
}  // namespace torch_xla
