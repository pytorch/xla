#ifndef XLA_TORCH_XLA_CSRC_OPS_NMS_H_
#define XLA_TORCH_XLA_CSRC_OPS_NMS_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Nms : public XlaNode {
 public:
  Nms(const torch::lazy::Value& boxes, const torch::lazy::Value& scores,
      const torch::lazy::Value& iou_threshold);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_NMS_H_
