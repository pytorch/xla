#pragma once

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class UpsampleBilinearBackward : public Node {
 public:
  UpsampleBilinearBackward(const Value& input, std::vector<int64_t> output_size,
                           std::vector<int64_t> input_size, bool align_corners);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& output_size() const { return output_size_; }

  const std::vector<int64_t>& input_size() const { return input_size_; }

  bool align_corners() const { return align_corners_; }

 private:
  std::vector<int64_t> output_size_;
  std::vector<int64_t> input_size_;
  bool align_corners_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
