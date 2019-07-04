#include "torch_xla/csrc/ops/mean.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
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

xla::XlaOp LowerMean(const xla::XlaOp& input,
                     const std::vector<xla::int64>& dimensions,
                     bool keep_reduced_dimensions,
                     const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildMean(input, dimensions, keep_reduced_dimensions);
  return dtype ? xla::ConvertElementType(
                     result, MakeXlaPrimitiveType(*dtype, /*device=*/nullptr))
               : result;
}

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<xla::int64>& dimensions,
                           bool keep_reduced_dimensions,
                           const c10::optional<at::ScalarType>& dtype) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return LowerMean(operands[0], dimensions, keep_reduced_dimensions, dtype);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Mean::Mean(const Value& input, std::vector<xla::int64> dimensions,
           bool keep_reduced_dimensions, c10::optional<at::ScalarType> dtype)
    : Node(
          ir::OpKind(at::aten::mean), {input},
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

NodePtr Mean::Clone(OpList operands) const {
  return MakeNode<Mean>(operands.at(0), dimensions_, keep_reduced_dimensions_,
                        dtype_);
}

XlaOpVector Mean::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(
      LowerMean(input, dimensions_, keep_reduced_dimensions_, dtype_), loctx);
}

std::string Mean::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=[" << absl::StrJoin(dimensions_, ", ")
     << "], keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
