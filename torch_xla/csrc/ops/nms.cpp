#include "torch_xla/csrc/ops/nms.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace {

xla::Shape NodeOutputShape(const torch::lazy::Value& boxes,
                           const torch::lazy::Value& scores,
                           const torch::lazy::Value& iou_threshold) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    return BuildNms(operands[0], operands[1], operands[2]);
  };

  return InferOutputShape(
      {GetXlaShape(boxes), GetXlaShape(scores), GetXlaShape(iou_threshold)},
      shape_fn);
}

}  // namespace

Nms::Nms(const torch::lazy::Value& boxes, const torch::lazy::Value& scores,
         const torch::lazy::Value& iou_threshold)
    : XlaNode(
          /*op=*/xla_nms,
          /*operands=*/{boxes, scores, iou_threshold},
          /*xla_shape_fn=*/
          [&]() { return NodeOutputShape(boxes, scores, iou_threshold); }) {}

torch::lazy::NodePtr Nms::Clone(torch::lazy::OpList operands) const {
  return torch_xla::MakeNode<Nms>(operands.at(0), operands.at(1),
                                  operands.at(2));
}

XlaOpVector Nms::Lower(LoweringContext* loctx) const {
  xla::XlaOp boxes = loctx->GetOutputOp(operand(0));
  xla::XlaOp scores = loctx->GetOutputOp(operand(1));
  xla::XlaOp iou_threshold = loctx->GetOutputOp(operand(2));
  return ReturnOp(BuildNms(boxes, scores, iou_threshold), loctx);
}

}  // namespace torch_xla
