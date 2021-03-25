#include "tensorflow/compiler/xla/literal.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"

namespace xla {

Literal::Literal(const Shape& shape) : shape_(shape) {
  std::vector<int64_t> dimensions = util::ToVector<int64_t>(shape.dimensions());
  at::TensorOptions options(
      static_cast<at::ScalarType>(PrimitiveToScalarType(shape.element_type())));
  value_ = at::empty(dimensions, options);
}

const Shape& Literal::shape() const { return shape_; }

}  // namespace xla
