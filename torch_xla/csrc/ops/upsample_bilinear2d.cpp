#include "torch_xla/csrc/ops/upsample_bilinear2d.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/resize_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

UpsampleBilinear::UpsampleBilinear(const Value& input,
                                   std::vector<xla::int64> output_size,
                                   bool align_corners)
    : Node(ir::OpKind(at::aten::upsample_bilinear2d), {input},
           [&]() {
             return resize::GetForwardOutputShape2d(input.shape(), output_size);
           },
           /*num_outputs=*/1, xla::util::MHash(output_size, align_corners)),
      output_size_(std::move(output_size)),
      align_corners_(align_corners) {}

NodePtr UpsampleBilinear::Clone(OpList operands) const {
  return MakeNode<UpsampleBilinear>(operands.at(0), output_size_,
                                    align_corners_);
}

XlaOpVector UpsampleBilinear::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output =
      resize::LowerForward2d("ResizeBilinear", input, shape(), align_corners_,
                             /*half_pixel_centers=*/!align_corners_);
  return ReturnOp(output, loctx);
}

std::string UpsampleBilinear::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=("
     << absl::StrJoin(output_size_, ", ")
     << "), align_corners=" << align_corners_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
