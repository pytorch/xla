#pragma once

#include "torch_xla/csrc/ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

// Generic IR Node implementation for nodes which can simply be described by a
// specific OpKind and a lowering function. IR nodes carrying metadata should
// not be using this class (and have the metadata captured by the LowerFn), but
// they should instead create a dedicated IR node. Doing the former would limit
// IR introspection.
class Generic : public Node {
 public:
  using LowerFn = std::function<XlaOpVector(const Node&, LoweringContext*)>;

  Generic(OpKind op, absl::Span<const Value> operands, xla::Shape shape,
          LowerFn lower_fn, size_t num_outputs = 1,
          xla::hash_t hash_seed = 0x5a2d296e9);

  Generic(OpKind op, absl::Span<const Value> operands,
          const std::function<xla::Shape()>& shape_fn, LowerFn lower_fn,
          size_t num_outputs = 1, xla::hash_t hash_seed = 0x5a2d296e9);

  Generic(OpKind op, xla::Shape shape, LowerFn lower_fn, size_t num_outputs,
          xla::hash_t hash_seed);

  NodePtr Clone(OpList operands) const override;

  XlaOpVector Lower(LoweringContext* loctx) const override;

 private:
  LowerFn lower_fn_;
  xla::hash_t hash_seed_;
};

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
