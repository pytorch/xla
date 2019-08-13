#include "torch_xla/csrc/ops/gather.h"

#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

bool IsSparseGather(const xla::XlaOp& input, const xla::XlaOp& index,
                    xla::int64 dim) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  xla::Shape index_shape = XlaHelpers::ShapeOfXlaOp(index);
  xla::int64 input_elements = xla::ShapeUtil::ElementsIn(input_shape);
  xla::int64 index_elements = xla::ShapeUtil::ElementsIn(index_shape);
  // Simple heuristic. Might need fine tuning.
  return index_elements < input_elements / 3;
}

xla::Shape NodeOutputShape(const Value& input, const Value& index,
                           xla::int64 dim) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    return TorchGather(operands[0], operands[1], dim,
                       IsSparseGather(operands[0], operands[1], dim));
  };
  return InferOutputShape({input.shape(), index.shape()}, lower_for_shape_fn);
}

}  // namespace

Gather::Gather(const Value& input, xla::int64 dim, const Value& index)
    : Node(ir::OpKind(at::aten::gather), {input, index},
           [&]() { return NodeOutputShape(input, index, dim); },
           /*num_outputs=*/1, xla::util::MHash(dim)),
      dim_(dim) {}

NodePtr Gather::Clone(OpList operands) const {
  return MakeNode<Gather>(operands.at(0), dim_, operands.at(1));
}

XlaOpVector Gather::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp index = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      TorchGather(input, index, dim_, IsSparseGather(input, index, dim_)),
      loctx);
}

std::string Gather::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
