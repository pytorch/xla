#include "torch_xla/csrc/ops/upsample_bilinear2d_backward.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

UpsampleBilinearBackward::UpsampleBilinearBackward(
    const Value& input, std::vector<xla::int64> output_size,
    std::vector<xla::int64> input_size, bool align_corners)
    : Node(ir::OpKind(at::aten::upsample_bilinear2d_backward), {input},
           [&]() {
             return resize::GetBackwardOutputShape2d(input.shape(), input_size);
           },
           /*num_outputs=*/1,
           xla::util::MHash(output_size, input_size, align_corners)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)),
      align_corners_(align_corners) {}

NodePtr UpsampleBilinearBackward::Clone(OpList operands) const {
  return MakeNode<UpsampleBilinearBackward>(operands.at(0), output_size_,
                                            input_size_, align_corners_);
}

XlaOpVector UpsampleBilinearBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerBackward2d(
      "ResizeBilinearGrad", input, shape(), align_corners_,
      /*half_pixel_centers=*/!align_corners_);
  return ReturnOp(output, loctx);
}

std::string UpsampleBilinearBackward::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ") << "), input_size=("
     << absl::StrJoin(input_size_, ", ")
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
