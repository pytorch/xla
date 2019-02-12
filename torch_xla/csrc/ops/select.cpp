#include "ops/select.h"

#include "helpers.h"
#include "lowering_context.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape GetSelectShape(const xla::Shape& input_shape, xla::int64 dim,
                          xla::int64 index) {
  auto new_dims = XlaHelpers::DropDimensions(input_shape.dimensions(), {dim});
  return xla::ShapeUtil::MakeShape(input_shape.element_type(), new_dims);
}

}  // namespace

Select::Select(const Value& input, xla::int64 dim, xla::int64 index)
    : Node(ir::OpKind(at::aten::select), {input},
           GetSelectShape(input.shape(), dim, index),
           /*num_outputs=*/1, xla::util::MHash(dim, index)),
      dim_(dim),
      index_(index) {}

XlaOpVector Select::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp slice = xla::SliceInDim(input, index_, index_ + 1, 1, dim_);
  xla::XlaOp output = xla::Reshape(slice, shape().dimensions());
  return ReturnOp(output, loctx);
}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", index=" << index_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
