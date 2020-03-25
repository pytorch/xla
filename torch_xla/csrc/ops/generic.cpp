#include "torch_xla/csrc/ops/generic.h"

#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Generic::Generic(OpKind op, absl::Span<const Value> operands, xla::Shape shape,
                 LowerFn lower_fn, size_t num_outputs, xla::hash_t hash_seed)
    : Node(std::move(op), operands, std::move(shape), num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, absl::Span<const Value> operands,
                 const std::function<xla::Shape()>& shape_fn, LowerFn lower_fn,
                 size_t num_outputs, xla::hash_t hash_seed)
    : Node(std::move(op), operands, shape_fn, num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

Generic::Generic(OpKind op, xla::Shape shape, LowerFn lower_fn,
                 size_t num_outputs, xla::hash_t hash_seed)
    : Node(std::move(op), std::move(shape), num_outputs, hash_seed),
      lower_fn_(std::move(lower_fn)),
      hash_seed_(hash_seed) {}

NodePtr Generic::Clone(OpList operands) const {
  return MakeNode<Generic>(op(), operands, shape(), lower_fn_, num_outputs(),
                           hash_seed_);
}

XlaOpVector Generic::Lower(LoweringContext* loctx) const {
  return lower_fn_(*this, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
