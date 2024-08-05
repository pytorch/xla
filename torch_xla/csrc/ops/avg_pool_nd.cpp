#include "torch_xla/csrc/ops/avg_pool_nd.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/shape_helper.h"

namespace torch_xla {
namespace {

// Infers the output shape of the max pooling operation.
xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           int64_t spatial_dim_count,
                           absl::Span<const int64_t> kernel_size,
                           absl::Span<const int64_t> stride,
                           absl::Span<const int64_t> padding, bool ceil_mode,
                           bool count_include_pad) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildAvgPoolNd(operands[0], spatial_dim_count, kernel_size, stride,
                          padding, ceil_mode, count_include_pad);
  };
  return InferOutputShape({GetXlaShape(input)}, lower_for_shape_fn);
}

c10::Symbol AvgPoolNdSymbol(int64_t spatial_dim_count) {
  switch (spatial_dim_count) {
    case 1:
      return at::aten::avg_pool1d;
    case 2:
      return at::aten::avg_pool2d;
    case 3:
      return at::aten::avg_pool3d;
    default:
      XLA_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

AvgPoolNd::AvgPoolNd(const torch::lazy::Value& input, int64_t spatial_dim_count,
                     std::vector<int64_t> kernel_size,
                     std::vector<int64_t> stride, std::vector<int64_t> padding,
                     bool ceil_mode, bool count_include_pad,
                     std::optional<int> divisor_override)
    : XlaNode(
          torch::lazy::OpKind(AvgPoolNdSymbol(spatial_dim_count)), {input},
          [&]() {
            return NodeOutputShape(input, spatial_dim_count, kernel_size,
                                   stride, padding, ceil_mode,
                                   count_include_pad);
          },
          /*num_outputs=*/1,
          torch::lazy::MHash(spatial_dim_count, kernel_size, stride, padding,
                             ceil_mode, count_include_pad)),
      spatial_dim_count_(spatial_dim_count),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      ceil_mode_(ceil_mode),
      count_include_pad_(count_include_pad),
      divisor_override_(divisor_override) {}

torch::lazy::NodePtr AvgPoolNd::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<AvgPoolNd>(operands.at(0), spatial_dim_count_,
                                        kernel_size_, stride_, padding_,
                                        ceil_mode_, count_include_pad_);
}

XlaOpVector AvgPoolNd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      BuildAvgPoolNd(input, spatial_dim_count_, kernel_size_, stride_, padding_,
                     ceil_mode_, count_include_pad_);
  if (divisor_override_) {
    auto dtype = ShapeHelper::ShapeOfXlaOp(output).element_type();
    auto* builder = loctx->builder();
    int size = std::accumulate(kernel_size_.begin(), kernel_size_.end(), 1,
                               std::multiplies<int>());
    output = xla::Div(
        xla::Mul(output, XlaHelpers::ScalarValue(size, dtype, builder)),
        XlaHelpers::ScalarValue(*divisor_override_, dtype, builder));
  }
  return ReturnOp(output, loctx);
}

std::string AvgPoolNd::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", spatial_dim_count=" << spatial_dim_count_
     << ", kernel_size=(" << absl::StrJoin(kernel_size_, ", ") << "), stride=("
     << absl::StrJoin(stride_, ", ") << "), padding=("
     << absl::StrJoin(padding_, ", ")
     << "), count_include_pad=" << count_include_pad_;
  return ss.str();
}

}  // namespace torch_xla
