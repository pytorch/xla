#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Nms : public Node {
 public:
  Nms(const Value& boxes, const Value& scores, const Value& score_threshold,
      const Value& iou_threshold, int64_t output_size);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t output_size() const { return output_size_; }

 private:
  int64_t output_size_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
