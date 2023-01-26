//#pragma once
//
//#include <vector>
//
//#include "torch_xla/csrc/ir.h"
//#include "torch_xla/csrc/torch_util.h"
//
//namespace torch_xla {
//
//class EmptySymInt : public XlaNode {
//  public:
//  EmptySymInt(const torch::lazy::Value& input,
//              const SymIntElements& size_elements);
//
//  std::string ToString() const override;
//
//  XlaOpVector Lower(LoweringContext* loctx) const override;
//
//  const std::vector<int64_t>& size() const { return upper_bounds_; }
//
//  const bool IsDynamic(int index) constr { return dynamic_dims_[index]; }
//
//  private:
//   std:vector<int64_t> upper_bounds_;
//   absl::InlinedVector<bool, xla::InlineRank()> dynamic_dims_;
//};
//
//}  // namespace torch_xla
//
