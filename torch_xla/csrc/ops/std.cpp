#include "torch_xla/csrc/ops/std.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/reduction.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           std::vector<xla::int64>& dimensions,
                           bool keep_reduced_dimensions,
                           xla::int64 correction) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildStdDeviation(operands[0], dimensions, keep_reduced_dimensions,
                             correction);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Std::Std(const Value& input, std::vector<xla::int64> dimensions,
         bool keep_reduced_dimensions, xla::int64 correction)
    : Node(ir::OpKind(at::aten::std), {input},
           [&]() {
             return NodeOutputShape(input, dimensions, keep_reduced_dimensions,
                                    correction);
           },
           /*num_outputs=*/1,
           xla::util::MHash(dimensions, keep_reduced_dimensions, correction)),
      dimensions_(std::move(dimensions)),
      keep_reduced_dimensions_(keep_reduced_dimensions),
      correction_(correction) {}

NodePtr Std::Clone(OpList operands) const {
  return MakeNode<Std>(operands.at(0), dimensions_, keep_reduced_dimensions_,
                       correction_);
}

XlaOpVector Std::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(BuildStdDeviation(input, dimensions_,
                                    keep_reduced_dimensions_, correction_),
                  loctx);
}

std::string Std::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << "), keep_reduced_dimensions=" << keep_reduced_dimensions_
     << ", correction=" << correction_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
