#include "torch_xla/csrc/ops/std_mean.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<int64_t>& dimensions,
                           bool keep_reduced_dimensions,
                           int64_t correction) {
  auto lower_for_shape_fn_std_mean =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp std = BuildStdDeviation(operands[0], dimensions,
                                       keep_reduced_dimensions, correction);
    xla::XlaOp mean =
        BuildMean(operands[0], dimensions, keep_reduced_dimensions);
    return xla::Tuple(operands[0].builder(), {std, mean});
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn_std_mean);
}

}  // namespace

StdMean::StdMean(const Value& input, std::vector<int64_t> dimensions,
                 int64_t correction, bool keep_reduced_dimensions)
    : Node(ir::OpKind(at::aten::std_mean), {input},
           [&]() {
             return NodeOutputShape(input, dimensions, keep_reduced_dimensions,
                                    correction);
           },
           /*num_outputs=*/2,
           torch::lazy::MHash(dimensions, correction, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

NodePtr StdMean::Clone(OpList operands) const {
  return MakeNode<StdMean>(operands.at(0), dimensions_, correction_,
                           keep_reduced_dimensions_);
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
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
