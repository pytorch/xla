#ifndef XLA_TORCH_XLA_CSRC_OPS_EIGH_H_
#define XLA_TORCH_XLA_CSRC_OPS_EIGH_H_

#include <string_view>

#include "torch_xla/csrc/ir.h"
#include "xla/types.h"

namespace torch_xla {

class Eigh : public XlaNode {
 public:
  Eigh(const torch::lazy::Value& input, std::string_view uplo);

  std::string ToString() const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  char uplo_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_EIGH_H_
