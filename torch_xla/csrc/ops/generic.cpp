#include "ops/generic.h"

#include "lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Generic::Generic(OpKind op, tensorflow::gtl::ArraySlice<const Value> operands,
                 xla::Shape shape, LowerFn lower_fn, size_t num_outputs)
    : Node(std::move(op), operands, std::move(shape), num_outputs),
      lower_fn_(std::move(lower_fn)) {}

XlaOpVector Generic::Lower(LoweringContext* loctx) const {
  return lower_fn_(*this, loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
