#ifndef XLA_TORCH_XLA_CSRC_OPS_UPDATE_SLICE_H_
#define XLA_TORCH_XLA_CSRC_OPS_UPDATE_SLICE_H_

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class UpdateSlice : public XlaNode {
 public:
  UpdateSlice(const torch::lazy::Value& input, const torch::lazy::Value& source,
              absl::Span<const int64_t> base_indices);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& base_indices() const { return base_indices_; }

 private:
  std::vector<int64_t> base_indices_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_UPDATE_SLICE_H_