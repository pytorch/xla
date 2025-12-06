#include "torch_xla/csrc/ops/upsample_nearest3d_backward.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"
#include "xla/util.h"

namespace torch_xla {

UpsampleNearest3dBackward::UpsampleNearest3dBackward(
    const torch::lazy::Value& input, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size)
    : XlaNode(
          torch::lazy::OpKind(at::aten::upsample_nearest3d_backward), {input},
          [&]() {
            return resize::GetBackwardOutputShape3d(GetXlaShape(input), input_size);
          },
          /*num_outputs=*/1,
          torch::lazy::MHash(output_size, input_size)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)) {}

torch::lazy::NodePtr UpsampleNearest3dBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<UpsampleNearest3dBackward>(operands.at(0),
                                                          output_size_,
                                                          input_size_);
}

XlaOpVector UpsampleNearest3dBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerBackward3d(
      input, xla_shape(), /*align_corners=*/false, /*half_pixel_centers=*/false);
  return ReturnOp(output, loctx);
}

std::string UpsampleNearest3dBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ")
     << "), input_size=(" << absl::StrJoin(input_size_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
