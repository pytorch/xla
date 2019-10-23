#include "torch_xla/csrc/ops/upsample_nearest2d.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(
    const Value& input,
    tensorflow::gtl::ArraySlice<const xla::int64> output_size) {
  XLA_CHECK_EQ(output_size.size(), 2);
  const xla::Shape& input_shape = input.shape();
  return xla::ShapeUtil::MakeShape(
      input_shape.element_type(),
      {input_shape.dimensions(0), input_shape.dimensions(1), output_size[0],
       output_size[1]});
}

std::string GetBackendConfig(bool align_corners, bool half_pixel_centers) {
  return absl::StrCat("\"", align_corners, half_pixel_centers, "\"");
}

xla::XlaOp LowerUpsampleNearest(const xla::XlaOp& input,
                                const xla::Shape& output_shape) {
  xla::Shape input_shape = XlaHelpers::ShapeOfXlaOp(input);
  if (input_shape.dimensions(2) == output_shape.dimensions(2) &&
      input_shape.dimensions(3) == output_shape.dimensions(3)) {
    return input;
  }
  if (input_shape.dimensions(2) == 1 && input_shape.dimensions(3) == 1) {
    return input + xla::Zeros(input.builder(), output_shape);
  }
  // XLA wants NHWC while PyTorch comes in as NCHW, so we need to transpose,
  // call the kernel, and transpose back.
  std::vector<xla::int64> transpose_permute({0, 3, 2, 1});
  auto inv_transpose_permute = xla::InversePermutation(transpose_permute);
  xla::Shape resized_shape =
      xla::ShapeUtil::PermuteDimensions(inv_transpose_permute, output_shape);
  xla::XlaOp tinput = xla::Transpose(input, transpose_permute);
  xla::XlaOp resised = xla::CustomCall(
      input.builder(), "ResizeNearest", {tinput}, resized_shape,
      GetBackendConfig(/*align_corners=*/false, /*half_pixel_centers=*/false));
  return xla::Transpose(resised, inv_transpose_permute);
}

}  // namespace

UpsampleNearest::UpsampleNearest(const Value& input,
                                 std::vector<xla::int64> output_size)
    : Node(ir::OpKind(at::aten::upsample_nearest2d), {input},
           [&]() { return NodeOutputShape(input, output_size); },
           /*num_outputs=*/1, xla::util::MHash(output_size)),
      output_size_(std::move(output_size)) {}

NodePtr UpsampleNearest::Clone(OpList operands) const {
  return MakeNode<UpsampleNearest>(operands.at(0), output_size_);
}

XlaOpVector UpsampleNearest::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = LowerUpsampleNearest(input, shape());
  return ReturnOp(output, loctx);
}

std::string UpsampleNearest::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
