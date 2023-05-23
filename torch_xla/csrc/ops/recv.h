#ifndef XLA_TORCH_XLA_CSRC_OPS_RECV_H_
#define XLA_TORCH_XLA_CSRC_OPS_RECV_H_

#include "torch_xla/csrc/cross_replica_reduces.h"
#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

class Recv : public XlaNode {
 public:
  Recv(const torch::lazy::Value& token, const xla::Shape& recv_shape,
       int64_t channel_id);

  std::string ToString() const override;

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

  int64_t channel_id() const { return channel_id_; }

 private:
  xla::Shape recv_shape_;
  int64_t channel_id_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_RECV_H_