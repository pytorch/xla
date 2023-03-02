#include "torch_xla/csrc/ops/upsample_bilinear2d_backward.h"

#include "absl/strings/str_join.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"

namespace torch_xla {

UpsampleBilinearBackward::UpsampleBilinearBackward(
    const torch::lazy::Value& input, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size, bool align_corners)
    : XlaNode(
          torch::lazy::OpKind(at::aten::upsample_bilinear2d_backward), {input},
          [&]() {
            return resize::GetBackwardOutputShape2d(GetXlaShape(input),
                                                    input_size);
          },
          /*num_outputs=*/1,
          torch::lazy::MHash(output_size, input_size, align_corners)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)),
      align_corners_(align_corners) {}

torch::lazy::NodePtr UpsampleBilinearBackward::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<UpsampleBilinearBackward>(
      operands.at(0), output_size_, input_size_, align_corners_);
}

XlaOpVector UpsampleBilinearBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerBackward2d(
      "ResizeBilinearGrad", input, xla_shape(), align_corners_,
      /*half_pixel_centers=*/!align_corners_);
  return ReturnOp(output, loctx);
}

std::string UpsampleBilinearBackward::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << "), input_size=("
     << absl::StrJoin(input_size_, ", ")
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace torch_xla
