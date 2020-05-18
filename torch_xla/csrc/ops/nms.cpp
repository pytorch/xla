#include "torch_xla/csrc/ops/nms.h"

#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/nms_op.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& boxes, const Value& scores,
                           const Value& score_threshold,
                           const Value& iou_threshold, xla::int64 output_size) {
  auto shape_fn = [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    NmsResult result = BuildNms(operands[0], operands[1], operands[2],
                                operands[3], output_size);
    return xla::Tuple(result.selected_indices.builder(),
                      {result.selected_indices, result.num_valid});
  };
  return InferOutputShape({boxes.shape(), scores.shape(),
                           score_threshold.shape(), iou_threshold.shape()},
                          shape_fn);
}

}  // namespace

Nms::Nms(const Value& boxes, const Value& scores, const Value& score_threshold,
         const Value& iou_threshold, xla::int64 output_size)
    : Node(xla_nms, {boxes, scores, score_threshold, iou_threshold},
           [&]() {
             return NodeOutputShape(boxes, scores, score_threshold,
                                    iou_threshold, output_size);
           },
           /*num_outputs=*/2, xla::util::MHash(output_size)),
      output_size_(output_size) {}

NodePtr Nms::Clone(OpList operands) const {
  return MakeNode<Nms>(operands.at(0), operands.at(1), operands.at(2),
                       operands.at(3), output_size_);
}

XlaOpVector Nms::Lower(LoweringContext* loctx) const {
  xla::XlaOp boxes = loctx->GetOutputOp(operand(0));
  xla::XlaOp scores = loctx->GetOutputOp(operand(1));
  xla::XlaOp score_threshold = loctx->GetOutputOp(operand(2));
  xla::XlaOp iou_threshold = loctx->GetOutputOp(operand(3));
  NmsResult result =
      BuildNms(boxes, scores, score_threshold, iou_threshold, output_size_);
  return ReturnOps({result.selected_indices, result.num_valid}, loctx);
}

std::string Nms::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", output_size=" << output_size_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
