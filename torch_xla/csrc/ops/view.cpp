#include "torch_xla/csrc/ops/view.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_sizes) {
  auto lower_for_shape_fn =
      [&output_sizes](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildView(operands[0], output_sizes);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

View::View(const Value& input, std::vector<xla::int64> output_size)
    : Node(ir::OpKind(at::aten::view), {input},
           [&]() { return NodeOutputShape(input, output_size); },
           /*num_outputs=*/1, xla::util::MHash(output_size)),
      output_size_(std::move(output_size)) {}

XlaOpVector View::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildView(input, output_size_);
  return ReturnOp(output, loctx);
}

std::string View::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
