#pragma once

#include <vector>

#include "absl/types/span.h"
#include "xla/shape.h"
#include "xla/types.h"

namespace torch_xla {

class ShapeBuilder {
 public:
  explicit ShapeBuilder(xla::PrimitiveType type) : type_(type) {}

  ShapeBuilder& Add(const xla::Shape& shape, int64_t dim);

  ShapeBuilder& Add(const xla::Shape& shape,
                    absl::Span<const int64_t> dimensions);

  ShapeBuilder& Add(int64_t size);

  xla::Shape Build() const;

 private:
  struct ShapeDim {
    const xla::Shape* shape = nullptr;
    int64_t dim_or_size = -1;
  };

  xla::PrimitiveType type_;
  std::vector<ShapeDim> dims_;
};

}  // namespace torch_xla
