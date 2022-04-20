#include "torch_xla/csrc/ops/upsample_nearest2d_backward.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

UpsampleNearestBackward::UpsampleNearestBackward(
    const Value& input, std::vector<int64_t> output_size,
    std::vector<int64_t> input_size)
    : Node(torch::lazy::OpKind(at::aten::upsample_nearest2d_backward), {input},
           [&]() {
             return resize::GetBackwardOutputShape2d(input.xla_shape(),
                                                     input_size);
           },
           /*num_outputs=*/1, torch::lazy::MHash(output_size, input_size)),
      output_size_(std::move(output_size)),
      input_size_(std::move(input_size)) {}

torch::lazy::NodePtr UpsampleNearestBackward::Clone(OpList operands) const {
  return ir::MakeNode<UpsampleNearestBackward>(operands.at(0), output_size_,
                                               input_size_);
}

XlaOpVector UpsampleNearestBackward::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = resize::LowerBackward2d(
      "ResizeNearestGrad", input, xla_shape(),
      /*align_corners=*/false, /*half_pixel_centers=*/false);
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
