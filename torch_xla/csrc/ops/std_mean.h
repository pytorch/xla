#ifndef XLA_TORCH_XLA_CSRC_OPS_STD_MEAN_H_
#define XLA_TORCH_XLA_CSRC_OPS_STD_MEAN_H_

#include <vector>

#include "torch_xla/csrc/ir.h"
#include "xla/types.h"

namespace torch_xla {

class StdMean : public XlaNode {
 public:
  StdMean(const torch::lazy::Value& input, std::vector<int64_t> dimensions,
          double correction, bool keep_reduced_dimensions);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  const std::vector<int64_t>& dimensions() const { return dimensions_; }

  double correction() const { return correction_; }

  bool keep_reduced_dimensions() const { return keep_reduced_dimensions_; }

 private:
  std::vector<int64_t> dimensions_;
  double correction_;
  bool keep_reduced_dimensions_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_STD_MEAN_H_