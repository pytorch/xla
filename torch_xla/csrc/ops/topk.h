#pragma once

#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {

class TopK : public XlaNode {
 public:
  TopK(const torch::lazy::Value& input, int64_t k, int64_t dim, bool largest,
       bool sorted, bool stable);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t k() const { return k_; };

  int64_t dim() const { return dim_; };

  bool largest() const { return largest_; }

  bool sorted() const { return sorted_; }

  bool stable() const { return stable_; }

 private:
  int64_t k_;
  int64_t dim_;
  bool largest_;
  bool sorted_;
  bool stable_;
};

class TopKSymInt : public XlaNode {
 public:
  TopKSymInt(const torch::lazy::Value& input, const SymIntElements& k,
             int64_t dim, bool largest, bool sorted, bool stable);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  int64_t k_upper_bound_;
  int64_t dim_;
  bool largest_;
  bool sorted_;
  bool stable_;
};

}  // namespace torch_xla
