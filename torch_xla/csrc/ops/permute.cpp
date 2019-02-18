#include "torch_xla/csrc/ops/permute.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           tensorflow::gtl::ArraySlice<const xla::int64> dims) {
  auto lower_for_shape_fn =
      [dims](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1);
    return xla::Transpose(operands[0], dims);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Permute::Permute(const Value& input,
                 tensorflow::gtl::ArraySlice<const xla::int64> dims)
    : Node(ir::OpKind(at::aten::permute), {input}, NodeOutputShape(input, dims),
           /*num_outputs=*/1, xla::util::MHash(dims)),
      dims_(dims.begin(), dims.end()) {}

XlaOpVector Permute::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::Transpose(input, dims_);
  return ReturnOp(output, loctx);
}

std::string Permute::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dims=[" << absl::StrJoin(dims_, ", ") << "]";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
