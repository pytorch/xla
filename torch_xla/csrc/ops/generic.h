#ifndef XLA_TORCH_XLA_CSRC_OPS_GENERIC_H_
#define XLA_TORCH_XLA_CSRC_OPS_GENERIC_H_

#include "torch_xla/csrc/ir.h"

namespace torch_xla {

// Generic IR XlaNode implementation for nodes which can simply be described by
// a specific OpKind and a lowering function. IR nodes carrying metadata should
// not be using this class (and have the metadata captured by the LowerFn), but
// they should instead create a dedicated IR node. Doing the former would limit
// IR introspection.
class Generic : public XlaNode {
 public:
  using LowerFn = std::function<XlaOpVector(const XlaNode&, LoweringContext*)>;

  Generic(torch::lazy::OpKind op, c10::ArrayRef<torch::lazy::Value> operands,
          xla::Shape shape, LowerFn lower_fn, size_t num_outputs = 1,
          torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9);

  Generic(torch::lazy::OpKind op, c10::ArrayRef<torch::lazy::Value> operands,
          const std::function<xla::Shape()>& shape_fn, LowerFn lower_fn,
          size_t num_outputs = 1,
          torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9);

  Generic(torch::lazy::OpKind op, c10::ArrayRef<torch::lazy::Value> operands,
          std::vector<torch::lazy::Shape>&& shapes,
          const std::function<xla::Shape()>& shape_fn, LowerFn lower_fn,
          size_t num_outputs = 1,
          torch::lazy::hash_t hash_seed = (uint32_t)0x5a2d296e9);

  Generic(torch::lazy::OpKind op, xla::Shape shape, LowerFn lower_fn,
          size_t num_outputs, torch::lazy::hash_t hash_seed);

  torch::lazy::NodePtr Clone(torch::lazy::OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  LowerFn lower_fn_;
  torch::lazy::hash_t hash_seed_;
};

}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_GENERIC_H_