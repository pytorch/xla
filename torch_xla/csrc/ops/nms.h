#ifndef XLA_TORCH_XLA_CSRC_OPS_NMS_H_
#define XLA_TORCH_XLA_CSRC_OPS_NMS_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Nms : public XlaNode {
 public:
  Nms(const torch::lazy::Value& boxes, const torch::lazy::Value& scores,
      const torch::lazy::Value& score_threshold,
      const torch::lazy::Value& iou_threshold, int64_t output_size);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t output_size() const { return output_size_; }

 private:
  int64_t output_size_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_NMS_H_
