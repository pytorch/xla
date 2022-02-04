#include "torch_xla/csrc/ops/var_mean.h"

#include "absl/strings/str_join.h"
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

xla::Shape NodeOutputShape(const Value& input, std::vector<int64_t>& dimensions,
                           int64_t correction, bool keep_reduced_dimensions) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    xla::XlaOp var =
        BuildVar(operands[0], dimensions, correction, keep_reduced_dimensions);
    xla::XlaOp mean =
        BuildMean(operands[0], dimensions, keep_reduced_dimensions);
    return xla::Tuple(operands[0].builder(), {var, mean});
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

VarMean::VarMean(const Value& input, std::vector<int64_t> dimensions,
                 int64_t correction, bool keep_reduced_dimensions)
    : Node(ir::OpKind(at::aten::var_mean), {input},
           [&]() {
             return NodeOutputShape(input, dimensions, correction,
                                    keep_reduced_dimensions);
           },
           /*num_outputs=*/2,
           torch::lazy::MHash(dimensions, correction, keep_reduced_dimensions)),
      dimensions_(std::move(dimensions)),
      correction_(correction),
      keep_reduced_dimensions_(keep_reduced_dimensions) {}

NodePtr VarMean::Clone(OpList operands) const {
  return MakeNode<VarMean>(operands.at(0), dimensions_, correction_,
                           keep_reduced_dimensions_);
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
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
