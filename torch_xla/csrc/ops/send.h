#ifndef XLA_TORCH_XLA_CSRC_OPS_SEND_H_
#define XLA_TORCH_XLA_CSRC_OPS_SEND_H_

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Send : public XlaNode {
 public:
  Send(const torch::lazy::Value& input, const torch::lazy::Value& token,
       int64_t channel_id);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t channel_id() const { return channel_id_; }

 private:
  int64_t channel_id_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_SEND_H_