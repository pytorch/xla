#pragma once

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch/csrc/jit/python/pybind.h"

namespace torch_xla {
namespace op_builder {

using BuilderPtr = std::shared_ptr<xla::XlaBuilder>;

struct Op {
  Op(BuilderPtr builder, xla::XlaOp op)
      : builder(std::move(builder)), op(std::move(op)) {}

  BuilderPtr builder;
  xla::XlaOp op;
};

using OpPtr = std::shared_ptr<Op>;

py::object ShapeToPyShape(const xla::Shape& shape);

xla::Shape PyShapeToShape(py::object shape);

OpPtr CreateOp(BuilderPtr builder, const std::string& opname,
               const std::vector<OpPtr>& operands, py::dict args);

}  // namespace op_builder
}  // namespace torch_xla
