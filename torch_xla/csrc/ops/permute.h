#ifndef XLA_TORCH_XLA_CSRC_OPS_PERMUTE_H_
#define XLA_TORCH_XLA_CSRC_OPS_PERMUTE_H_

#include "absl/types/span.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class Permute : public XlaNode {
 public:
  Permute(const torch::lazy::Value& input, std::vector<int64_t> dims);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& dims() const { return dims_; }

  static xla::Shape MakePermuteShape(const xla::Shape& source_shape,
                                     absl::Span<const int64_t> permutation);

 private:
  // The permutation of dimensions.
  std::vector<int64_t> dims_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_PERMUTE_H_