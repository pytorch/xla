#include "torch_xla/csrc/ops/prod.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerProd(xla::XlaOp input,
                     const std::vector<xla::int64>& dimensions,
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

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<xla::int64>& dimensions,
                           bool keep_reduced_dimensions,
                           c10::optional<at::ScalarType> dtype) {
  xla::Shape input_shape;
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::Shape input_shape;
    xla::XlaOp output = LowerProd(
        XlaHelpers::MaybeMakeArray(operands[0], dimensions, &input_shape),
        dimensions, keep_reduced_dimensions, dtype);
    return XlaHelpers::MaybeReshapeToScalar(output, input_shape);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Prod::Prod(const Value& input, std::vector<xla::int64> dimensions,
           bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::prod), {input},
           [&]() {
             return NodeOutputShape(input, dimensions, keep_reduced_dimensions,
                                    dtype);
           },
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, keep_reduced_dimensions,
                            OptionalOr<int>(dtype, -1))),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      dtype_(dtype) {}

NodePtr Prod::Clone(OpList operands) const {
  return MakeNode<Prod>(operands.at(0), dimensions_, keep_reduced_dimensions_,
                        dtype_);
}

XlaOpVector Prod::Lower(LoweringContext* loctx) const {
  xla::Shape input_shape;
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      LowerProd(XlaHelpers::MaybeMakeArray(input, dimensions_, &input_shape),
                dimensions_, keep_reduced_dimensions_, dtype_);
  return ReturnOp(XlaHelpers::MaybeReshapeToScalar(output, input_shape), loctx);
}

std::string Prod::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
