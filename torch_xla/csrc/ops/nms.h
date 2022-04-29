#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Nms : public XlaNode {
 public:
  Nms(const XlaValue& boxes, const XlaValue& scores,
      const XlaValue& score_threshold, const XlaValue& iou_threshold,
      int64_t output_size);

  torch::lazy::NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t output_size() const { return output_size_; }

 private:
  int64_t output_size_;
};

}  // namespace torch_xla
