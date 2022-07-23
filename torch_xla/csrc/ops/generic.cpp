#include "torch_xla/csrc/ops/generic.h"

#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {

Generic::Generic(torch::lazy::OpKind op,
                 c10::ArrayRef<torch::lazy::Value> operands, xla::Shape shape,
                 LowerFn lower_fn, size_t num_outputs,
                 torch::lazy::hash_t hash_seed)
    : XlaNode(std::move(op), operands, std::move(shape), num_outputs,
              hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

Generic::Generic(torch::lazy::OpKind op,
                 c10::ArrayRef<torch::lazy::Value> operands,
                 const std::function<xla::Shape()>& shape_fn, LowerFn lower_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : XlaNode(std::move(op), operands, shape_fn, num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

Generic::Generic(torch::lazy::OpKind op,
                 c10::ArrayRef<torch::lazy::Value> operands,
                 std::vector<torch::lazy::Shape>&& shapes,
                 const std::function<xla::Shape()>& shape_fn, LowerFn lower_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : XlaNode(std::move(op), operands, std::move(shapes), shape_fn, num_outputs,
              hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

Generic::Generic(torch::lazy::OpKind op, xla::Shape shape, LowerFn lower_fn,
                 size_t num_outputs, torch::lazy::hash_t hash_seed)
    : XlaNode(std::move(op), std::move(shape), num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

torch::lazy::NodePtr Generic::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Generic>(op(), operands, xla_shape(), lower_fn_,
                                        num_outputs(), hash_seed_);
}

XlaOpVector Generic::Lower(LoweringContext* loctx) const {
  return lower_fn_(*this, loctx);
}

}  // namespace torch_xla
