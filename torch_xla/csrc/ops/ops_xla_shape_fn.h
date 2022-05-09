#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {

xla::Shape AbsOutputShape(const XlaValue& input);

xla::Shape MaximumOutputShape(const XlaValue& input, const XlaValue& other);

}  // namespace torch_xla
