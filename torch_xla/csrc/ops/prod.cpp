#include "torch_xla/csrc/ops/prod.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerProd(xla::XlaOp input, const std::vector<int64_t>& dimensions,
                     bool keep_reduced_dimensions,
                     c10::optional<at::ScalarType> dtype) {
  xla::XlaOp casted_input;
  if (dtype) {
    casted_input = ConvertTo(input, XlaHelpers::TypeOfXlaOp(input),
                             MakeXlaPrimitiveType(*dtype, /*device=*/nullptr),
                             /*device=*/nullptr);
  } else {
    casted_input = ConvertToNumeric(input, XlaHelpers::TypeOfXlaOp(input));
  }
  return BuildProd(casted_input, dimensions, keep_reduced_dimensions);
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           std::vector<int64_t>& dimensions,
                           bool keep_reduced_dimensions,
                           c10::optional<at::ScalarType> dtype) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return LowerProd(operands[0], dimensions, keep_reduced_dimensions, dtype);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Prod::Prod(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
           bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype)
    : XlaNode(torch::lazy::OpKind(at::aten::prod), {input},
              [&]() {
                return NodeOutputShape(input, dimensions,
                                       keep_reduced_dimensions, dtype);
              },
              /*num_outputs=*/1,
              torch::lazy::MHash(dimensions, keep_reduced_dimensions,
                                 torch::lazy::OptionalOr<int>(dtype, -1))),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      dtype_(dtype) {}

torch::lazy::NodePtr Prod::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Prod>(operands.at(0), dimensions_,
                                     keep_reduced_dimensions_, dtype_);
}

XlaOpVector Prod::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(
      LowerProd(input, dimensions_, keep_reduced_dimensions_, dtype_), loctx);
}

std::string Prod::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dimensions=("
     << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", dtype=" << torch::lazy::OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace torch_xla
