#include "torch_xla/csrc/ops/std_mean.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           std::vector<int64_t>& dimensions,
                           bool keep_reduced_dimensions, double correction) {
  auto lower_for_shape_fn_std_mean =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp std = BuildStdDeviation(operands[0], dimensions,
                                       keep_reduced_dimensions, correction);
    xla::XlaOp mean =
        BuildMean(operands[0], dimensions, keep_reduced_dimensions);
    return xla::Tuple(operands[0].builder(), {std, mean});
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn_std_mean);
}

}  // namespace

StdMean::StdMean(const torch::lazy::Value& input,
                 std::vector<int64_t> dimensions, double correction,
                 bool keep_reduced_dimensions)
    : XlaNode(
          torch::lazy::OpKind(at::aten::std_mean), {input},
          [&]() {
            return NodeOutputShape(input, dimensions, keep_reduced_dimensions,
                                   correction);
          },
          /*num_outputs=*/2,
          torch::lazy::MHash(dimensions, correction, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

torch::lazy::NodePtr StdMean::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<StdMean>(operands.at(0), dimensions_,
                                        correction_, keep_reduced_dimensions_);
}

XlaOpVector StdMean::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp op_std = BuildStdDeviation(input, dimensions_,
                                        keep_reduced_dimensions_, correction_);
  xla::XlaOp op_mean = BuildMean(input, dimensions_, keep_reduced_dimensions_);
  return ReturnOps({op_std, op_mean}, loctx);
}

std::string StdMean::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dimensions=("
     << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace torch_xla
