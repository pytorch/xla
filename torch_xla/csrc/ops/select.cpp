#include "torch_xla/csrc/ops/select.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

Select::Select(const Value& input, int64_t dim, int64_t start, int64_t end,
               int64_t stride)
    : Node(xla_select, {input},
           [&]() {
             return MakeSelectShape(input.shape(), dim, start, end, stride);
           },
           /*num_outputs=*/1, torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

NodePtr Select::Clone(OpList operands) const {
  return MakeNode<Select>(operands.at(0), dim_, start_, end_, stride_);
}

XlaOpVector Select::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::SliceInDim(input, start_, end_,
                                      GetStride(start_, end_, stride_), dim_);
  return ReturnOp(output, loctx);
}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_ << ", start=" << start_
     << ", end=" << end_ << ", stride=" << stride_;
  return ss.str();
}

xla::Shape Select::MakeSelectShape(const xla::Shape& shape, int64_t dim,
                                   int64_t start, int64_t end, int64_t stride) {
  int64_t effective_stride = GetStride(start, end, stride);
  xla::Shape select_shape(shape);
  select_shape.set_dimensions(
      dim, (end - start + effective_stride - 1) / effective_stride);
  return select_shape;
}

int64_t Select::GetStride(int64_t start, int64_t end, int64_t stride) {
  if (stride == 0) {
    XLA_CHECK_EQ(start, end);
    stride = 1;
  }
  return stride;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
