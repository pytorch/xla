#ifndef XLA_TORCH_XLA_CSRC_OPS_MEAN_H_
#define XLA_TORCH_XLA_CSRC_OPS_MEAN_H_

#include <c10/core/ScalarType.h>

#include <optional>
#include <vector>

#include "torch_xla/csrc/ir.h"
#include "xla/types.h"

namespace torch_xla {

class Mean : public XlaNode {
 public:
  Mean(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
       bool keep_reduced_dimensions, std::optional<at::ScalarType> dtype);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

  const std::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  std::vector<int64_t> dimensions_;
  bool keep_reduced_dimensions_;
  std::optional<at::ScalarType> dtype_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_MEAN_H_
