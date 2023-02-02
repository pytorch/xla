#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class ScatterReduce : public XlaNode {
 public:
  ScatterReduce(const torch::lazy::Value& input,
                const torch::lazy::Value& index, const torch::lazy::Value& src,
                c10::string_view reduce, bool include_self, int64_t dim);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t dim() const { return dim_; };

 private:
  c10::string_view reduce_;
  bool include_self_;
  int64_t dim_;
};

}  // namespace torch_xla
