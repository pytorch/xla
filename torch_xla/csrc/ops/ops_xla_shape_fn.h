#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {

xla::Shape AbsOutputShape(const XlaValue& input);

xla::Shape AcosOutputShape(const XlaValue& input);

xla::Shape AcoshOutputShape(const XlaValue& input);

xla::Shape AsinOutputShape(const XlaValue& input);

xla::Shape AsinhOutputShape(const XlaValue& input);

xla::Shape AtanOutputShape(const XlaValue& input);

xla::Shape AtanhOutputShape(const XlaValue& input);

xla::Shape CosOutputShape(const XlaValue& input);

xla::Shape CoshOutputShape(const XlaValue& input);
xla::Shape LogOutputShape(const XlaValue& input);

xla::Shape Log2OutputShape(const XlaValue& input);

xla::Shape Log10OutputShape(const XlaValue& input);

xla::Shape MaximumOutputShape(const XlaValue& input, const XlaValue& other);

xla::Shape SgnOutputShape(const XlaValue& input);

xla::Shape SignOutputShape(const XlaValue& input);

}  // namespace torch_xla
