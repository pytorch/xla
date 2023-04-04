#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Multinomial : public XlaNode {
 public:
  Multinomial(const torch::lazy::Value& input,
              const torch::lazy::Value& seed,
              int64_t num_samples, bool replacement);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t num_samples_;
  bool replacement_;
};

}  // namespace torch_xla
