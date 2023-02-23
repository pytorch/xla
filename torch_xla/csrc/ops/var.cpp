#include "torch_xla/csrc/ops/var.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           std::vector<int64_t>& dimensions, double correction,
                           bool keep_reduced_dimensions) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildVar(operands[0], dimensions, correction,
                    keep_reduced_dimensions);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

Var::Var(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
         double correction, bool keep_reduced_dimensions)
    : XlaNode(
          torch::lazy::OpKind(at::aten::var), {input},
          NodeOutputShape(input, dimensions, correction,
                          keep_reduced_dimensions),
          /*num_outputs=*/1,
          torch::lazy::MHash(dimensions, correction, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

torch::lazy::NodePtr Var::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Var>(operands.at(0), dimensions_, correction_,
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
  ss << XlaNode::ToString() << ", dimensions=("
     << absl::StrJoin(dimensions_, ", ") << "), correction=" << correction_
     << ", keep_reduced_dimensions=" << keep_reduced_dimensions_;
  return ss.str();
}

}  // namespace torch_xla
