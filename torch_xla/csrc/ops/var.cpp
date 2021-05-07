#include "torch_xla/csrc/ops/var.h"

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
                           xla::int64 correction,
                           bool keep_reduced_dimensions) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildVar(operands[0], dimensions, correction,
                    keep_reduced_dimensions);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Var::Var(const Value& input, std::vector<xla::int64> dimensions,
         xla::int64 correction, bool keep_reduced_dimensions)
    : Node(ir::OpKind(at::aten::var), {input},
           NodeOutputShape(input, dimensions, correction,
                           keep_reduced_dimensions),
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, correction, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

NodePtr Var::Clone(OpList operands) const {
  return MakeNode<Var>(operands.at(0), dimensions_, correction_,
                       keep_reduced_dimensions_);
}

XlaOpVector Var::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(
      BuildVar(input, dimensions_, correction_, keep_reduced_dimensions_),
      loctx);
}

std::string Var::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), correction=" << correction_
     << ", keep_reduced_dimensions=" << keep_reduced_dimensions_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
