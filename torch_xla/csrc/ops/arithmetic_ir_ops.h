#ifndef XLA_TORCH_XLA_CSRC_OPS_ARITHMETIC_IR_OPS_H_
#define XLA_TORCH_XLA_CSRC_OPS_ARITHMETIC_IR_OPS_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

torch::lazy::NodePtr operator+(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);
torch::lazy::NodePtr operator-(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);
torch::lazy::NodePtr operator*(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);
torch::lazy::NodePtr operator/(const torch::lazy::Value& node1,
                               const torch::lazy::Value& node2);

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_ARITHMETIC_IR_OPS_H_