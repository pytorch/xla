#pragma once

#include <ostream>
#include <vector>

#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "lazy_xla/csrc/compiler/debug_macros.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {

class Shape {
 public:
  Shape() : element_type_(PrimitiveType::INVALID) {}

  Shape(PrimitiveType element_type, absl::Span<const int64> dimensions)
      : element_type_(element_type),
        dimensions_(dimensions.begin(), dimensions.end()) {}

  Shape(absl::Span<const Shape> element_shapes)
      : element_type_(PrimitiveType::TUPLE),
        element_shapes_(element_shapes.begin(), element_shapes.end()) {}

  std::string ToString(bool print_layout = false) const {
    return absl::StrCat(PrimitiveTypeName(element_type_), "[",
                        absl::StrJoin(dimensions_, ","), "]");
  }

  int64 rank() const { return dimensions_.size(); }

  bool IsArray() const { return false; }

  bool IsTuple() const { return element_type_ == PrimitiveType::TUPLE; }

  bool is_static() const { TF_LOG(FATAL) << "Not implemented yet."; }

  bool is_dynamic_dimension(int dimension) const { return false; }

  void set_dynamic_dimension(int dimension, bool is_dynamic) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  absl::Span<const bool> dynamic_dimensions() const {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  void DeleteDimension(int64 dim_to_delete) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  PrimitiveType element_type() const { return element_type_; }
  void set_element_type(PrimitiveType value) { element_type_ = value; }

  int64 dimensions(int index) const {
    XLA_CHECK_LT(index, dimensions_.size());
    return dimensions_[index];
  }

  void set_dimensions(int index, int64 value) {
    XLA_CHECK_LT(index, dimensions_.size());
    dimensions_[index] = value;
  }
  void add_dimensions(int64 value) { TF_LOG(FATAL) << "Not implemented yet."; }

  absl::Span<const int64> dimensions() const {
    return absl::MakeSpan(dimensions_);
  }
  absl::Span<int64> mutable_dimensions() {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  int tuple_shapes_size() const { return element_shapes_.size(); }

  const Shape& tuple_shapes(int index) const {
    XLA_CHECK_GE(index, 0);
    XLA_CHECK_LT(index, element_shapes_.size());
    return element_shapes_[index];
  }
  const std::vector<Shape>& tuple_shapes() const { return element_shapes_; }

  const Layout& layout() const { TF_LOG(FATAL) << "Not implemented yet."; }

  bool operator==(const Shape& other) const {
    return element_type_ == other.element_type_ &&
           dimensions_ == other.dimensions_;
  }

 private:
  PrimitiveType element_type_;
  std::vector<int64> dimensions_;
  std::vector<Shape> element_shapes_;
};

class ProgramShape {
 public:
  ProgramShape() : parameters_size_(0){};

  ProgramShape(Shape shape, int parameters_size)
      : shape_(std::move(shape)), parameters_size_(parameters_size) {}

  int parameters_size() const { return parameters_size_; }

  const Shape& result() const { return shape_; }

 private:
  Shape shape_;
  int parameters_size_;
};

inline std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  return out << shape.ToString();
}

}  // namespace xla
