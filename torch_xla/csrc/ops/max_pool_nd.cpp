#include "torch_xla/csrc/ops/max_pool_nd.h"

#include "xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           int64_t spatial_dim_count,
                           absl::Span<const int64_t> kernel_size,
                           absl::Span<const int64_t> stride,
                           absl::Span<const int64_t> padding, bool ceil_mode) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    MaxPoolResult result =
        BuildMaxPoolNd(operands[0], spatial_dim_count, kernel_size, stride,
                       padding, ceil_mode);
    return xla::Tuple(operands[0].builder(), {result.result, result.indices});
  };
  return InferOutputShape({GetXlaShape(input)}, shape_fn);
}

c10::Symbol MaxPoolNdSymbol(int64_t spatial_dim_count) {
  switch (spatial_dim_count) {
    case 1:
      return at::aten::max_pool1d;
    case 2:
      return at::aten::max_pool2d;
    case 3:
      return at::aten::max_pool3d;
    default:
      XLA_ERROR() << "Invalid number of spatial dimensions: "
                  << spatial_dim_count;
  }
}

}  // namespace

MaxPoolNd::MaxPoolNd(const torch::lazy::Value& input, int64_t spatial_dim_count,
                     std::vector<int64_t> kernel_size,
                     std::vector<int64_t> stride, std::vector<int64_t> padding,
                     bool ceil_mode)
    : XlaNode(torch::lazy::OpKind(MaxPoolNdSymbol(spatial_dim_count)), {input},
              [&]() {
                return NodeOutputShape(input, spatial_dim_count, kernel_size,
                                       stride, padding, ceil_mode);
              },
              /*num_outputs=*/2,
              torch::lazy::MHash(spatial_dim_count, kernel_size, stride,
                                 padding, ceil_mode)),
      spatial_dim_count_(spatial_dim_count),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      ceil_mode_(ceil_mode) {}

torch::lazy::NodePtr MaxPoolNd::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<MaxPoolNd>(operands.at(0), spatial_dim_count_,
                                          kernel_size_, stride_, padding_,
                                          ceil_mode_);
}

XlaOpVector MaxPoolNd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  MaxPoolResult result = BuildMaxPoolNd(input, spatial_dim_count_, kernel_size_,
                                        stride_, padding_, ceil_mode_);
  return ReturnOps({result.result, result.indices}, loctx);
}

std::string MaxPoolNd::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", spatial_dim_count=" << spatial_dim_count_
     << ", kernel_size=(" << absl::StrJoin(kernel_size_, ", ") << "), stride=("
     << absl::StrJoin(stride_, ", ") << "), padding=("
     << absl::StrJoin(padding_, ", ") << "), ceil_mode=" << ceil_mode_;
  return ss.str();
}

}  // namespace torch_xla
