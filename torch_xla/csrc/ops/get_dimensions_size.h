#ifndef XLA_TORCH_XLA_CSRC_OPS_GET_DIMENSIONS_SIZE_H_
#define XLA_TORCH_XLA_CSRC_OPS_GET_DIMENSIONS_SIZE_H_

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class GetDimensionsSize : public XlaNode {
 public:
  GetDimensionsSize(const torch::lazy::Value& input,
                    std::vector<int64_t> dimensions);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

 private:
  std::vector<int64_t> dimensions_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_GET_DIMENSIONS_SIZE_H_