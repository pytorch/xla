#include "torch_xla/csrc/ops/any.h"

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

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<xla::int64>& dimensions,
                           bool keep_reduced_dimensions) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return BuildAny(operands[0], dimensions, keep_reduced_dimensions);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Any::Any(const Value& input, std::vector<xla::int64> dimensions,
         bool keep_reduced_dimensions)
    : Node(ir::OpKind(at::aten::any), {input},
           NodeOutputShape(input, dimensions, keep_reduced_dimensions),
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

NodePtr Any::Clone(OpList operands) const {
  return MakeNode<Any>(operands.at(0), dimensions_, keep_reduced_dimensions_);
}

XlaOpVector Any::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildAny(input, dimensions_, keep_reduced_dimensions_),
                  loctx);
}

std::string Any::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=[" << absl::StrJoin(dimensions_, ", ")
     << "], keep_reduced_dimensions=" << keep_reduced_dimensions_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
