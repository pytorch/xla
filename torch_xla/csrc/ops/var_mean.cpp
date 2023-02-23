#include "torch_xla/csrc/ops/var_mean.h"

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
    xla::XlaOp var =
        BuildVar(operands[0], dimensions, correction, keep_reduced_dimensions);
    xla::XlaOp mean =
        BuildMean(operands[0], dimensions, keep_reduced_dimensions);
    return xla::Tuple(operands[0].builder(), {var, mean});
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

}  // namespace

VarMean::VarMean(const torch::lazy::Value& input,
                 std::vector<int64_t> dimensions, double correction,
                 bool keep_reduced_dimensions)
    : XlaNode(
          torch::lazy::OpKind(at::aten::var_mean), {input},
          [&]() {
            return NodeOutputShape(input, dimensions, correction,
                                   keep_reduced_dimensions);
          },
          /*num_outputs=*/2,
          torch::lazy::MHash(dimensions, correction, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

torch::lazy::NodePtr VarMean::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<VarMean>(operands.at(0), dimensions_,
                                        correction_, keep_reduced_dimensions_);
}

XlaOpVector VarMean::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp op_var =
      BuildVar(input, dimensions_, correction_, keep_reduced_dimensions_);
  xla::XlaOp op_mean = BuildMean(input, dimensions_, keep_reduced_dimensions_);
  return ReturnOps({op_var, op_mean}, loctx);
}

std::string VarMean::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dimensions=("
     << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace torch_xla
