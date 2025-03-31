#include "torch_xla/csrc/ops/select.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {

Select::Select(const torch::lazy::Value& input, int64_t dim, int64_t start,
               int64_t end, int64_t stride)
    : XlaNode(
          xla_select, {input},
          [&]() {
            return MakeSelectShape(GetXlaShape(input), dim, start, end, stride);
          },
          /*num_outputs=*/1, torch::lazy::MHash(dim, start, end, stride)),
      dim_(dim),
      start_(start),
      end_(end),
      stride_(stride) {}

torch::lazy::NodePtr Select::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<Select>(operands.at(0), dim_, start_, end_,
                                     stride_);
}

XlaOpVector Select::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  const xla::Shape& input_shape = ShapeHelper::ShapeOfXlaOp(input);
  if (!input_shape.is_unbounded_dynamic()) {
    xla::XlaOp output = xla::SliceInDim(input, start_, end_,
                                        GetStride(start_, end_, stride_), dim_);
    return ReturnOp(output, loctx);
  } else {
    // When input has unbounded dynamic dim and target dim is the unbounded
    // dim, slice full range along the dynamic dim. We will assert now.
    std::vector<int32_t> start_vec(input_shape.dimensions_size(), 0);
    start_vec[dim_] = start_;
    xla::XlaOp starts =
        xla::ConstantR1(input.builder(), absl::Span<const int32_t>(start_vec));
    std::vector<int32_t> stride_vec(input_shape.dimensions_size(), 1);
    stride_vec[dim_] = GetStride(start_, end_, stride_);
    xla::XlaOp strides =
        xla::ConstantR1(input.builder(), absl::Span<const int32_t>(stride_vec));
    xla::Shape final_shape =
        MakeSelectShape(input_shape, dim_, start_, end_, stride_);
    std::vector<xla::XlaOp> limit_ops;
    for (int i = 0; i < input_shape.dimensions_size(); ++i) {
      if (input_shape.is_unbounded_dynamic_dimension(i)) {
        limit_ops.push_back(xla::Reshape(xla::GetDimensionSize(input, i), {1}));
        final_shape.set_unbounded_dynamic_dimension(i);
      } else {
        int32_t limit = i == dim_ ? end_ : input_shape.dimensions(i);
        limit_ops.push_back(xla::ConstantR1(
            input.builder(), absl::Span<const int32_t>({limit})));
      }
    }
    xla::XlaOp concat_limit = xla::ConcatInDim(input.builder(), limit_ops, {0});
    xla::XlaOp output =
        xla::CustomCall(input.builder(), "mhlo.real_dynamic_slice",
                        /*operands=*/{input, starts, concat_limit, strides},
                        /*shape*/ final_shape);
    return ReturnOp(output, loctx);
  }
}

std::string Select::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_ << ", start=" << start_
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

}  // namespace torch_xla
