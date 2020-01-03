#pragma once

#include <vector>

#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace torch_xla {

class ShapeBuilder {
 public:
  explicit ShapeBuilder(xla::PrimitiveType type) : type_(type) {}

  ShapeBuilder& Add(const xla::Shape& shape, xla::int64 dim);

  ShapeBuilder& Add(const xla::Shape& shape,
                    tensorflow::gtl::ArraySlice<const xla::int64> dimensions);

  ShapeBuilder& Add(xla::int64 size);

  xla::Shape Build() const;

 private:
  struct ShapeDim {
    const xla::Shape* shape = nullptr;
    xla::int64 dim_or_size = -1;
  };

  xla::PrimitiveType type_;
  std::vector<ShapeDim> dims_;
};

}  // namespace torch_xla
