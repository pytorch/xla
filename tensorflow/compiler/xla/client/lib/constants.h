#pragma once

#include "lazy_xla/csrc/compiler/helpers.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

// Returns scalar 'value' as a scalar of 'type'. Unlike ConstantR0, 'type' is
// determined at C++ run-time, rather than C++ compile-time.

template <typename T>
XlaOp ConstantR0WithType(XlaBuilder* builder, PrimitiveType type, T value) {
  if (std::is_floating_point<T>::value &&
      !(primitive_util::IsFloatingPointType(type) ||
        primitive_util::IsComplexType(type))) {
    TF_LOG(FATAL) << "Invalid cast from floating point type to "
                  << PrimitiveTypeName(type) << " in ConstantR0WithType.";
  }
  if (std::is_same<T, complex64>::value &&
      !primitive_util::IsComplexType(type)) {
    TF_LOG(FATAL) << "Invalid cast from complex type to "
                  << PrimitiveTypeName(type) << " in ConstantR0WithType.";
  }
  switch (type) {
    case PrimitiveType::PRED:
      return ConstantR0<bool>(builder, static_cast<bool>(value));
    case PrimitiveType::F16:
      return ConstantR0<half>(builder, static_cast<half>(value));
    case PrimitiveType::BF16:
      return ConstantR0<bfloat16>(builder, static_cast<bfloat16>(value));
    case PrimitiveType::F32:
      return ConstantR0<float>(builder, static_cast<float>(value));
    case PrimitiveType::F64:
      return ConstantR0<double>(builder, static_cast<double>(value));
    case PrimitiveType::C64:
      return ConstantR0<complex64>(builder, static_cast<complex64>(value));
    case PrimitiveType::C128:
      return ConstantR0<complex128>(builder, static_cast<complex128>(value));
    case PrimitiveType::U8:
      return ConstantR0<uint8>(builder, static_cast<uint8>(value));
    case PrimitiveType::U16:
      return ConstantR0<uint16>(builder, static_cast<uint16>(value));
    case PrimitiveType::U32:
      return ConstantR0<uint32>(builder, static_cast<uint32>(value));
    case PrimitiveType::U64:
      return ConstantR0<uint64>(builder, static_cast<uint64>(value));
    case PrimitiveType::S8:
      return ConstantR0<int8>(builder, static_cast<int8>(value));
    case PrimitiveType::S16:
      return ConstantR0<int16>(builder, static_cast<int16>(value));
    case PrimitiveType::S32:
      return ConstantR0<int32>(builder, static_cast<int32>(value));
    case PrimitiveType::S64:
      return ConstantR0<int64>(builder, static_cast<int64>(value));
    default:
      TF_LOG(FATAL) << "Invalid type for ConstantR0WithType ("
                    << PrimitiveTypeName(type) << ").";
  }
}

// Returns a scalar containing 'value' cast to the same run-time type as
// 'prototype'.
// If 'value' is floating point but 'prototype' is not, or if 'value' is
// complex 'prototype' is not, an error will be returned.
template <typename T>
XlaOp ScalarLike(XlaOp prototype, T value) {
  XlaBuilder* builder = prototype.builder();
  const Shape& shape =
      torch_lazy_tensors::compiler::XlaHelpers::ShapeOfXlaOp(prototype);
  return ConstantR0WithType(builder, shape.element_type(), value);
}

XlaOp Zero(XlaBuilder* builder, PrimitiveType type);

inline XlaOp Zeros(XlaBuilder* builder, const Shape& shape) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp ZerosLike(XlaOp prototype) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

XlaOp One(XlaBuilder* builder, PrimitiveType type);

inline XlaOp MinValue(XlaBuilder* builder, PrimitiveType type) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp MaxValue(XlaBuilder* builder, PrimitiveType type) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

inline XlaOp NanValue(XlaBuilder* builder, PrimitiveType type) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

}  // namespace xla
