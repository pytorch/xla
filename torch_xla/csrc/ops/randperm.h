#ifndef XLA_TORCH_XLA_CSRC_OPS_RANDPERM_H_
#define XLA_TORCH_XLA_CSRC_OPS_RANDPERM_H_

#include <vector>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

class RandPerm : public XlaNode {
 public:
  RandPerm(int64_t n, const at::ScalarType dtype, const at::Layout layout,
           const at::Device device, bool pin_memory);

  XlaOpVector Lower(LoweringContext* loctx) const override;
  std::string ToString() const override;

 private:
  int64_t n_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_RANDPERM_H_
