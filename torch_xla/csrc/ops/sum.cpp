#include "torch_xla/csrc/ops/sum.h"

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

xla::XlaOp LowerSum(const xla::XlaOp& input,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
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
  return BuildSum(casted_input, dimensions, keep_reduced_dimensions);
}

xla::Shape NodeOutputShape(
    const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions,
    bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return LowerSum(operands[0], dimensions, keep_reduced_dimensions, dtype);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Sum::Sum(const Value& input, std::vector<xla::int64> dimensions,
         bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::sum), {input},
           NodeOutputShape(input, dimensions, keep_reduced_dimensions, dtype),
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, keep_reduced_dimensions,
                            OptionalOr<int>(dtype, -1))),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      dtype_(dtype) {}

XlaOpVector Sum::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(
      LowerSum(input, dimensions_, keep_reduced_dimensions_, dtype_), loctx);
}

std::string Sum::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=[" << absl::StrJoin(dimensions_, ", ")
     << "], keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
