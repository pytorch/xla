#include "torch_xla/csrc/ops/avg_pool_nd.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/pooling.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

// Infers the output shape of the max pooling operation.
xla::Shape NodeOutputShape(
    const Value& input, xla::int64 spatial_dim_count,
    tensorflow::gtl::ArraySlice<const xla::int64> kernel_size,
    tensorflow::gtl::ArraySlice<const xla::int64> stride,
    tensorflow::gtl::ArraySlice<const xla::int64> padding,
    bool count_include_pad) {
  auto lower_for_shape_fn =
      [&](tensorflow::gtl::ArraySlice<const xla::XlaOp> operands)
      -> xla::XlaOp {
    XLA_CHECK_EQ(operands.size(), 1)
        << "Unexpected number of operands: " << operands.size();
    return BuildAvgPoolNd(operands[0], spatial_dim_count, kernel_size, stride,
                          padding, count_include_pad);
  };
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

c10::Symbol AvgPoolNdSymbol(xla::int64 spatial_dim_count) {
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

AvgPoolNd::AvgPoolNd(const Value& input, xla::int64 spatial_dim_count,
                     std::vector<xla::int64> kernel_size,
                     std::vector<xla::int64> stride,
                     std::vector<xla::int64> padding, bool count_include_pad)
    : Node(
          ir::OpKind(AvgPoolNdSymbol(spatial_dim_count)), {input},
          [&]() {
            return NodeOutputShape(input, spatial_dim_count, kernel_size,
                                   stride, padding, count_include_pad);
          },
          /*num_outputs=*/1,
          xla::util::MHash(spatial_dim_count, kernel_size, stride, padding,
                           count_include_pad)),
      spatial_dim_count_(spatial_dim_count),
      kernel_size_(std::move(kernel_size)),
      stride_(std::move(stride)),
      padding_(std::move(padding)),
      count_include_pad_(count_include_pad) {}

XlaOpVector AvgPoolNd::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildAvgPoolNd(input, spatial_dim_count_, kernel_size_,
                                     stride_, padding_, count_include_pad_);
  return ReturnOp(output, loctx);
}

std::string AvgPoolNd::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", spatial_dim_count=" << spatial_dim_count_
     << ", kernel_size=[" << absl::StrJoin(kernel_size_, ", ") << "], stride=["
     << absl::StrJoin(stride_, ", ") << "], padding=["
     << absl::StrJoin(padding_, ", ")
     << "], count_include_pad=" << count_include_pad_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
