#include "torch_xla/csrc/ops/upsample_nearest2d_backward.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> input_size) {
  return xla::ShapeUtil::MakeShape(input.shape().element_type(), input_size);
}

std::string GetBackendConfig(bool align_corners, bool half_pixel_centers) {
  return absl::StrCat("\"", align_corners, half_pixel_centers, "\"");
}

double ResizeFactor(const xla::Shape& input_shape,
                    const xla::Shape& output_shape, int dim) {
  return static_cast<double>(input_shape.dimensions(dim)) /
         static_cast<double>(output_shape.dimensions(dim));
}

xla::XlaOp LowerUpsampleNearestBackward(const xla::XlaOp& input,
                                        const xla::Shape& output_shape) {
  static double resiple_split_factor =
      xla::sys_util::GetEnvDouble("XLA_RESIZE_SPLIT_FACTOR", 3.0);
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  if (input_shape.dimensions(2) == output_shape.dimensions(2) &&
      input_shape.dimensions(3) == output_shape.dimensions(3)) {
    return input;
  }
  // XLA wants NHWC while PyTorch comes in as NCHW, so we need to transpose,
  // call the kernel, and transpose back.
  std::vector<xla::int64> transpose_permute({0, 3, 2, 1});
  auto inv_transpose_permute = xla::InversePermutation(transpose_permute);
  xla::Shape resized_shape =
      xla::ShapeUtil::PermuteDimensions(inv_transpose_permute, output_shape);
  xla::XlaOp tinput = xla::Transpose(input, transpose_permute);
  std::string backend_config =
      GetBackendConfig(/*align_corners=*/false, /*half_pixel_centers=*/false);
  if (ResizeFactor(input_shape, output_shape, 2) > resiple_split_factor &&
      ResizeFactor(input_shape, output_shape, 3) > resiple_split_factor) {
    // If the resize is too large, do one dimension at a time.
    xla::Shape partial_shape = resized_shape;
    // Partial shape is in NHWC, while input shape is in NCHW.
    partial_shape.mutable_dimensions()[1] = input_shape.dimensions(2);
    tinput = xla::CustomCall(input.builder(), "ResizeNearestGrad", {tinput},
                             partial_shape, backend_config);
  }
  xla::XlaOp resised = xla::CustomCall(input.builder(), "ResizeNearestGrad",
                                       {tinput}, resized_shape, backend_config);
  return xla::Transpose(resised, inv_transpose_permute);
}

}  // namespace

UpsampleNearestBackward::UpsampleNearestBackward(
    const Value& input, std::vector<xla::int64> output_size,
    std::vector<xla::int64> input_size)
    : Node(ir::OpKind(at::aten::upsample_nearest2d_backward), {input},
           [&]() { return NodeOutputShape(input, input_size); },
           /*num_outputs=*/1, xla::util::MHash(output_size, input_size)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)) {}

NodePtr UpsampleNearestBackward::Clone(OpList operands) const {
  return MakeNode<UpsampleNearestBackward>(operands.at(0), output_size_,
                                           input_size_);
}

XlaOpVector UpsampleNearestBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = LowerUpsampleNearestBackward(input, shape());
  return ReturnOp(output, loctx);
}

std::string UpsampleNearestBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << "), input_size=("
     << absl::StrJoin(input_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
