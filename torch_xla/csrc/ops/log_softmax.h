#ifndef XLA_TORCH_XLA_CSRC_OPS_LOG_SOFTMAX_H_
#define XLA_TORCH_XLA_CSRC_OPS_LOG_SOFTMAX_H_

#include <c10/core/ScalarType.h>
#include <c10/util/Optional.h>

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// IR node for log(softmax) operation.
class LogSoftmax : public XlaNode {
 public:
  LogSoftmax(const torch::lazy::Value& input, int64_t dim,
             c10::optional<at::ScalarType> dtype,
             std::vector<torch::lazy::Shape>&& shapes);

  XlaOpVector Lower(LoweringContext* loctx) const override;

  std::string ToString() const override;

  int64_t dim() const { return dim_; }

  const c10::optional<at::ScalarType>& dtype() const { return dtype_; }

 private:
  // The dimension along which the result is computed.
  int64_t dim_;
  c10::optional<at::ScalarType> dtype_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_LOG_SOFTMAX_H_