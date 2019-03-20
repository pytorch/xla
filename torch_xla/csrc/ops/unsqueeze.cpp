#include "torch_xla/csrc/ops/unsqueeze.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input, int dim) {
  auto lower_for_shape_fn =
      [dim](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp { return BuildUnsqueeze(operands[0], dim); };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Unsqueeze::Unsqueeze(const Value& input, int dim)
    : Node(
          ir::OpKind(at::aten::squeeze), {input},
          [&]() { return NodeOutputShape(input, dim); },
          /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

XlaOpVector Unsqueeze::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildUnsqueeze(input, dim_);
  return ReturnOp(output, loctx);
}

std::string Unsqueeze::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << " dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
