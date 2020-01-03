#include "torch_xla/csrc/shape_builder.h"

#include "tensorflow/compiler/xla/shape_util.h"

namespace torch_xla {

ShapeBuilder& ShapeBuilder::Add(const xla::Shape& shape, xla::int64 dim) {
  dims_.push_back({&shape, dim});
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(
    const xla::Shape& shape,
    tensorflow::gtl::ArraySlice<const xla::int64> dimensions) {
  dims_.reserve(dimensions.size());
  for (auto dim : dimensions) {
    dims_.push_back({&shape, dim});
  }
  return *this;
}

ShapeBuilder& ShapeBuilder::Add(xla::int64 size) {
  dims_.push_back({nullptr, size});
  return *this;
}

xla::Shape ShapeBuilder::Build() const {
  std::vector<xla::int64> dimensions;
  dimensions.reserve(dims_.size());
  for (auto& sdim : dims_) {
    if (sdim.shape != nullptr) {
      dimensions.push_back(sdim.shape->dimensions(sdim.dim_or_size));
    } else {
      dimensions.push_back(sdim.dim_or_size);
    }
  }
  xla::Shape shape = xla::ShapeUtil::MakeShape(type_, dimensions);
  for (xla::int64 i = 0; i < shape.rank(); ++i) {
    const ShapeDim& sdim = dims_[i];
    if (sdim.shape != nullptr) {
      shape.set_dynamic_dimension(
          i, sdim.shape->is_dynamic_dimension(sdim.dim_or_size));
    }
  }
  return shape;
}

}  // namespace torch_xla
