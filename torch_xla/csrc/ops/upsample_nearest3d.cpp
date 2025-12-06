#include "torch_xla/csrc/ops/upsample_nearest3d.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"
#include "xla/util.h"

namespace torch_xla {

UpsampleNearest3d::UpsampleNearest3d(const torch::lazy::Value& input,
                                     std::vector<int64_t> output_size)
    : XlaNode(
          torch::lazy::OpKind(at::aten::upsample_nearest3d), {input},
          [&]() {
            return resize::GetForwardOutputShape3d(GetXlaShape(input), output_size);
          },
          /*num_outputs=*/1, torch::lazy::MHash(output_size)),
      output_size_(std::move(output_size)) {}

torch::lazy::NodePtr UpsampleNearest3d::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<UpsampleNearest3d>(operands.at(0), output_size_);
}

XlaOpVector UpsampleNearest3d::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerForward3d(
      input, xla_shape(), /*align_corners=*/false, /*half_pixel_centers=*/false);
  return ReturnOp(output, loctx);
}

std::string UpsampleNearest3d::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
