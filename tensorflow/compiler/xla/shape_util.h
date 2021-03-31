#pragma once

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "lazy_tensors/computation_client/util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "torch/csrc/jit/tensorexpr/types.h"

namespace xla {

class ShapeIndex {
 public:
  ShapeIndex(std::initializer_list<int64> init) : indices_(init) {}

  bool empty() const { return indices_.empty(); }
  size_t size() const { return indices_.size(); }

  const int64& operator[](size_t i) const { return indices_[i]; }
  int64& operator[](size_t i) { return indices_[i]; }

 private:
  std::vector<int64> indices_;
};

class ShapeUtil {
 public:
  static int64 ElementsIn(const Shape& shape) {
    return lazy_tensors::util::Multiply<xla::int64>(shape.dimensions());
  }

  static int64 ByteSizeOfPrimitiveType(PrimitiveType primitive_type) {
    switch (primitive_type) {
      case PrimitiveType::PRED:
        return sizeof(int8);
      case PrimitiveType::S8:
        return sizeof(int8);
      case PrimitiveType::S16:
        return sizeof(int16);
      case PrimitiveType::S32:
        return sizeof(int32);
      case PrimitiveType::S64:
        return sizeof(int64);
      case PrimitiveType::U8:
        return sizeof(uint8);
      case PrimitiveType::U16:
        return sizeof(uint16);
      case PrimitiveType::U32:
        return sizeof(uint32);
      case PrimitiveType::U64:
        return sizeof(uint64);
      case PrimitiveType::BF16:
        return sizeof(float) / 2;
      case PrimitiveType::F16:
        return sizeof(float) / 2;
      case PrimitiveType::F32:
        return sizeof(float);
      case PrimitiveType::F64:
        return sizeof(double);
      case PrimitiveType::C64:
        return sizeof(complex64);
      case PrimitiveType::C128:
        return sizeof(complex128);
      default:
        TF_LOG(FATAL) << "Unhandled primitive type " << primitive_type;
    }
  }

  static bool SameDimensions(const Shape& lhs, const Shape& rhs) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  static bool SameElementType(const Shape& lhs, const Shape& rhs) {
    return lhs.element_type() == rhs.element_type();
  }

  static bool Compatible(const Shape& lhs, const Shape& rhs) {
    return lhs == rhs;
  }

  static Shape ChangeElementType(const Shape& original, PrimitiveType type) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  static Shape MakeTupleShape(absl::Span<const Shape> shapes) {
    return Shape(shapes);
  }

  static void AppendMajorDimension(int bound, Shape* shape) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  static Shape MakeShape(PrimitiveType element_type,
                         absl::Span<const int64> dimensions) {
    return MakeShapeWithDescendingLayout(element_type, dimensions);
  }

  static Shape MakeShapeWithLayout(PrimitiveType element_type,
                                   absl::Span<const int64> dimensions,
                                   absl::Span<const int64> minor_to_major,
                                   absl::Span<const Tile> tiles = {},
                                   int64 element_size_in_bits = 0,
                                   int64 memory_space = 0) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  static Shape MakeShapeWithDescendingLayout(
      PrimitiveType element_type, absl::Span<const int64> dimensions) {
    return Shape(element_type, dimensions);
  }

  static const Shape& GetTupleElementShape(const Shape& shape, int64 index) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }

  using MutatingVisitorFunction =
      std::function<void(Shape* /*subshape*/, const ShapeIndex& /*index*/)>;
  static void ForEachMutableSubshape(Shape* shape,
                                     const MutatingVisitorFunction& func) {
    if (!shape->IsTuple()) {
      return;
    }
    for (size_t i = 0; i < shape->tuple_shapes_size(); ++i) {
      func(shape, {static_cast<int64>(i)});
    }
  }

  static bool ElementIsIntegral(const Shape& shape) {
    return primitive_util::IsIntegralType(shape.element_type());
  }

  static Shape PermuteDimensions(absl::Span<const int64> permutation,
                                 const Shape& shape) {
    TF_LOG(FATAL) << "Not implemented yet.";
  }
};

inline torch::jit::tensorexpr::ScalarType PrimitiveToScalarType(
    xla::PrimitiveType scalar_type) {
  switch (scalar_type) {
    case xla::PrimitiveType::S8: {
      return torch::jit::tensorexpr::ScalarType::Char;
    }
    case xla::PrimitiveType::S16: {
      return torch::jit::tensorexpr::ScalarType::Short;
    }
    case xla::PrimitiveType::S32: {
      return torch::jit::tensorexpr::ScalarType::Int;
    }
    case xla::PrimitiveType::S64: {
      return torch::jit::tensorexpr::ScalarType::Long;
    }
    case xla::PrimitiveType::U8: {
      return torch::jit::tensorexpr::ScalarType::Byte;
    }
    case xla::PrimitiveType::F32: {
      return torch::jit::tensorexpr::ScalarType::Float;
    }
    case xla::PrimitiveType::F64: {
      return torch::jit::tensorexpr::ScalarType::Double;
    }
    case xla::PrimitiveType::PRED: {
      return torch::jit::tensorexpr::ScalarType::Bool;
    }
    default: { TF_LOG(FATAL) << "Not implemented yet."; }
  }
}

}  // namespace xla
