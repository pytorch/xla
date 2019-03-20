#include "torch_xla/csrc/ops/einsum.h"

#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(tensorflow::gtl::ArraySlice<const ir::Value> values,
                           const std::string& equation) {
  XLA_CHECK_EQ(values.size(), 2)
      << "Only two inputs supported for einsum for now";
  auto lower_for_shape_fn =
      [equation](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return xla::Einsum(operands[0], operands[1], equation,
                       XlaHelpers::mat_mul_precision());
  };
  std::vector<xla::Shape> shapes;
  shapes.reserve(values.size());
  for (auto& value : values) {
    shapes.push_back(value.shape());
  }
  return InferOutputShape(shapes, lower_for_shape_fn);
}

}  // namespace

Einsum::Einsum(const std::string& equation,
               tensorflow::gtl::ArraySlice<const ir::Value> values)
    : Node(
          ir::OpKind(at::aten::einsum), values,
          [&]() { return NodeOutputShape(values, equation); },
          /*num_outputs=*/1, xla::util::MHash(equation)),
      equation_(equation) {}

XlaOpVector Einsum::Lower(LoweringContext* loctx) const {
  xla::XlaOp output = xla::Einsum(loctx->GetOutputOp(operand(0)),
                                  loctx->GetOutputOp(operand(1)), equation_,
                                  XlaHelpers::mat_mul_precision());
  return ReturnOp(output, loctx);
}

std::string Einsum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", equation=" << equation_;
  return ss.str();
}

bool Einsum::SupportsEquation(const std::string& equation) {
  auto einsum_config_numeric_or_err = xla::ParseEinsumString(equation);
  if (!einsum_config_numeric_or_err.ok()) {
    return false;
  }
  auto einsum_config_numeric = einsum_config_numeric_or_err.ConsumeValueOrDie();
  auto validation_status = xla::ValidateEinsumNumericDimensions(
      /*x_config=*/einsum_config_numeric[0],
      /*y_config=*/einsum_config_numeric[1],
      /*output_config=*/einsum_config_numeric[2]);
  return validation_status.ok();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
