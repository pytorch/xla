#include "torch_xla/csrc/random.h"

#include <string>
#include <tuple>

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/prng.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/compiler/xla/xla_client/sys_util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/device.h"
#include "torch_xla/csrc/helpers.h"

namespace torch_xla {
namespace {

std::string GetDefaultGitGeneratorName() {
  XlaDeviceType hw_type = static_cast<XlaDeviceType>(GetCurrentDevice().type());
  switch (hw_type) {
    case XlaDeviceType::GPU:
      return "three_fry";
    default:
      return "default";
  }
}

xla::BitGeneratorTy GetBitGenerator() {
  static const std::string* bit_generator =
      new std::string(xla::sys_util::GetEnvString(
          "XLA_RNG_BIT_GENERATOR", GetDefaultGitGeneratorName()));
  if (*bit_generator == "default") {
    return [](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
      state = xla::ConcatScalars(key.builder(), {key, state});
      xla::XlaOp result =
          xla::RngBitGenerator(xla::RandomAlgorithm::RNG_DEFAULT, state, shape);
      return xla::RngOutput{/*value=*/xla::GetTupleElement(result, 1),
                            /*state=*/xla::GetTupleElement(result, 0)};
    };
  } else if (*bit_generator == "philox") {
    return [](xla::XlaOp key, xla::XlaOp state, const xla::Shape& shape) {
      std::tie(state, key) = xla::ScramblePhiloxKey(key);
      return xla::PhiloxBitGenerator(key, state, shape);
    };
  } else if (*bit_generator == "three_fry") {
    return xla::ThreeFryBitGenerator;
  }
  XLA_ERROR() << "Unknow random bit generator: " << *bit_generator;
}

xla::XlaOp MakeSeed(xla::XlaOp seed) {
  const xla::Shape& rng_shape = XlaHelpers::ShapeOfXlaOp(seed);
  if (rng_shape.element_type() == xla::PrimitiveType::U64) {
    return seed;
  }
  return xla::ConvertElementType(seed, xla::PrimitiveType::U64);
}

xla::XlaOp MakeUniformBoundaryValue(xla::XlaOp val) {
  xla::PrimitiveType element_type = XlaHelpers::TypeOfXlaOp(val);
  if (element_type == xla::PrimitiveType::BF16 ||
      element_type == xla::PrimitiveType::F16) {
    return xla::ConvertElementType(val, xla::PrimitiveType::F32);
  } else if (xla::primitive_util::IsComplexType(element_type)) {
    return xla::Real(val);
  }
  return val;
}

xla::Shape MakeRngShape(const xla::Shape& shape) {
  xla::PrimitiveType element_type = shape.element_type();
  xla::Shape rng_shape(shape);
  if (element_type == xla::PrimitiveType::BF16 ||
      element_type == xla::PrimitiveType::F16) {
    rng_shape.set_element_type(xla::PrimitiveType::F32);
  } else if (xla::primitive_util::IsComplexType(element_type)) {
    rng_shape.set_element_type(
        xla::primitive_util::ComplexComponentType(element_type));
  }
  return rng_shape;
}

}  // namespace

xla::XlaOp RngDiscreteUniform(xla::XlaOp seed, const xla::Shape& shape,
                              xla::XlaOp minval, xla::XlaOp maxval) {
  xla::PrimitiveType minval_type = XlaHelpers::TypeOfXlaOp(minval);
  xla::PrimitiveType maxval_type = XlaHelpers::TypeOfXlaOp(maxval);
  XLA_CHECK_EQ(minval_type, maxval_type);
  XLA_CHECK(minval_type == xla::PrimitiveType::S64 ||
            minval_type == xla::PrimitiveType::U64 ||
            minval_type == xla::PrimitiveType::S32 ||
            minval_type == xla::PrimitiveType::U32)
      << "RngDiscreteUniform not implemented for type "
      << xla::primitive_util::LowercasePrimitiveTypeName(minval_type);
  xla::XlaOp rng_seed = MakeSeed(seed);
  xla::XlaOp initial_state =
      xla::Zero(rng_seed.builder(), xla::PrimitiveType::U64);
  xla::Shape rng_shape(shape);
  rng_shape.set_element_type(minval_type);
  xla::XlaOp result =
      xla::UniformIntDistribution(rng_seed, initial_state, GetBitGenerator(),
                                  minval, maxval, rng_shape)
          .value;
  return MaybeConvertTo(result, shape.element_type());
}

xla::XlaOp RngUniform(xla::XlaOp seed, const xla::Shape& shape,
                      xla::XlaOp minval, xla::XlaOp maxval) {
  xla::XlaOp rng_seed = MakeSeed(seed);
  xla::Shape rng_shape = MakeRngShape(shape);
  xla::XlaOp rng_minval = MakeUniformBoundaryValue(minval);
  xla::XlaOp rng_maxval = MakeUniformBoundaryValue(maxval);
  xla::XlaOp initial_state =
      xla::Zero(rng_seed.builder(), xla::PrimitiveType::U64);
  switch (shape.element_type()) {
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::BF16: {
      xla::XlaOp rng = xla::UniformFloatingPointDistribution(
                           rng_seed, initial_state, GetBitGenerator(),
                           rng_minval, rng_maxval, rng_shape)
                           .value;
      return xla::ConvertElementType(rng, shape.element_type());
    }
    case xla::PrimitiveType::F32:
    case xla::PrimitiveType::F64:
      return xla::UniformFloatingPointDistribution(
                 rng_seed, initial_state, GetBitGenerator(), rng_minval,
                 rng_maxval, rng_shape)
          .value;
    case xla::PrimitiveType::C64:
    case xla::PrimitiveType::C128: {
      xla::XlaOp k_seed = XlaHelpers::ScalarValue<uint64_t>(
          17, XlaHelpers::TypeOfXlaOp(rng_seed), rng_seed.builder());
      xla::XlaOp rng_real = xla::UniformFloatingPointDistribution(
                                rng_seed, initial_state, GetBitGenerator(),
                                rng_minval, rng_maxval, rng_shape)
                                .value;
      xla::XlaOp rng_imag =
          xla::UniformFloatingPointDistribution(
              rng_seed * k_seed, initial_state, GetBitGenerator(), rng_minval,
              rng_maxval, rng_shape)
              .value;
      return xla::Complex(rng_real, rng_imag);
    }
    case xla::PrimitiveType::S32:
    case xla::PrimitiveType::U32:
    case xla::PrimitiveType::S64:
    case xla::PrimitiveType::U64:
      return xla::UniformIntDistribution(rng_seed, initial_state,
                                         GetBitGenerator(), rng_minval,
                                         rng_maxval, rng_shape)
          .value;
    default:
      XLA_ERROR() << "RngUniform not implemented for type "
                  << xla::primitive_util::LowercasePrimitiveTypeName(
                         shape.element_type());
  }
}

xla::XlaOp RngNormal(xla::XlaOp seed, const xla::Shape& shape, xla::XlaOp mean,
                     xla::XlaOp std) {
  xla::XlaOp rng_seed = MakeSeed(seed);
  xla::Shape rng_shape = MakeRngShape(shape);
  xla::XlaOp initial_state =
      xla::Zero(rng_seed.builder(), xla::PrimitiveType::U64);
  switch (shape.element_type()) {
    case xla::PrimitiveType::F16:
    case xla::PrimitiveType::BF16: {
      xla::XlaOp f32_mean = MaybeConvertTo(mean, xla::PrimitiveType::F32);
      xla::XlaOp f32_std = MaybeConvertTo(std, xla::PrimitiveType::F32);
      xla::XlaOp rng =
          xla::NormalFloatingPointDistribution(rng_seed, initial_state,
                                               GetBitGenerator(), rng_shape)
              .value;
      return xla::ConvertElementType(f32_mean + rng * f32_std,
                                     shape.element_type());
    }
    case xla::PrimitiveType::F32:
    case xla::PrimitiveType::F64: {
      xla::XlaOp rng =
          xla::NormalFloatingPointDistribution(rng_seed, initial_state,
                                               GetBitGenerator(), rng_shape)
              .value;
      return XlaHelpers::PromotedAdd(mean, XlaHelpers::PromotedMul(rng, std));
    }
    case xla::PrimitiveType::C64:
    case xla::PrimitiveType::C128: {
      xla::XlaOp k_seed = XlaHelpers::ScalarValue<uint64_t>(
          17, XlaHelpers::TypeOfXlaOp(rng_seed), rng_seed.builder());
      xla::XlaOp rng_real =
          xla::NormalFloatingPointDistribution(rng_seed, initial_state,
                                               GetBitGenerator(), rng_shape)
              .value;
      xla::XlaOp rng_imag =
          xla::NormalFloatingPointDistribution(rng_seed * k_seed, initial_state,
                                               GetBitGenerator(), rng_shape)
              .value;
      xla::XlaOp rng = xla::Complex(rng_real, rng_imag);
      // Variance for normal distribution of the real and imaginary values is
      // half of the input variance.
      xla::XlaOp sqrtTwo = XlaHelpers::ScalarValue(
          std::sqrt(2), XlaHelpers::TypeOfXlaOp(std), rng_seed.builder());
      return XlaHelpers::PromotedAdd(
          mean, XlaHelpers::PromotedMul(rng, std / sqrtTwo));
    }
    default:
      XLA_ERROR() << "RngNormal not implemented for type "
                  << xla::primitive_util::LowercasePrimitiveTypeName(
                         shape.element_type());
  }
}

}  // namespace torch_xla
