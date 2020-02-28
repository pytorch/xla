#include "torch_xla/csrc/random.h"

#include <string>
#include <tuple>

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"

namespace torch_xla {
namespace {

xla::BitGeneratorTy GetBitGenerator() {
  static const std::string* bit_generator = new std::string(
      xla::sys_util::GetEnvString("XLA_RNG_BIT_GENERATOR", "philox"));
  if (*bit_generator == "philox") {
    return [](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
      std::tie(state, key) = xla::ScramblePhiloxKey(key);
      return xla::PhiloxBitGenerator(key, state, shape);
    };
  } else if (*bit_generator == "three_fry") {
    return xla::ThreeFryBitGenerator;
  }
  XLA_ERROR() << "Unknow random bit generator: " << *bit_generator;
}

}  // namespace

xla::XlaOp RngUniform(xla::XlaOp seed, const xla::Shape& shape,
                      xla::XlaOp minval, xla::XlaOp maxval) {
  xla::XlaOp initial_state = xla::Zero(seed.builder(), xla::PrimitiveType::U64);
  switch (shape.element_type()) {
    case xla::PrimitiveType::BF16: {
      xla::Shape f32_shape(shape);
      f32_shape.set_element_type(xla::PrimitiveType::F32);
      xla::XlaOp f32_minval =
          xla::ConvertElementType(minval, xla::PrimitiveType::F32);
      xla::XlaOp f32_maxval =
          xla::ConvertElementType(maxval, xla::PrimitiveType::F32);
      xla::XlaOp rng = xla::UniformFloatingPointDistribution(
                           seed, initial_state, GetBitGenerator(), f32_minval,
                           f32_maxval, f32_shape)
                           .value;
      return xla::ConvertElementType(rng, xla::PrimitiveType::BF16);
    }
    case xla::PrimitiveType::F32:
    case xla::PrimitiveType::F64:
      return xla::UniformFloatingPointDistribution(
                 seed, initial_state, GetBitGenerator(), minval, maxval, shape)
          .value;
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U64:
      return xla::UniformIntDistribution(seed, initial_state, GetBitGenerator(),
                                         minval, maxval, shape)
          .value;
    default:
      XLA_ERROR() << "RngUniform not implemented for type "
                  << xla::primitive_util::LowercasePrimitiveTypeName(
                         shape.element_type());
  }
}

}  // namespace torch_xla
