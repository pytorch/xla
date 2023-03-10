#ifndef XLA_TORCH_XLA_CSRC_OPS_GENERIC_SLICE_H_
#define XLA_TORCH_XLA_CSRC_OPS_GENERIC_SLICE_H_

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class GenericSlice : public XlaNode {
 public:
  GenericSlice(const torch::lazy::Value& input,
               absl::Span<const int64_t> base_indices,
               absl::Span<const int64_t> sizes);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const { return base_indices_; }

  const std::vector<int64_t>& sizes() const { return sizes_; }

 private:
  std::vector<int64_t> base_indices_;
  std::vector<int64_t> sizes_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_GENERIC_SLICE_H_