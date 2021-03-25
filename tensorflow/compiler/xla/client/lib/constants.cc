#include "tensorflow/compiler/xla/client/lib/constants.h"

#include "tensorflow/compiler/xla/literal.h"

namespace xla {

XlaOp Zero(XlaBuilder* builder, PrimitiveType type) {
  Literal zero(ShapeUtil::MakeShape(type, {}));
  switch (type) {
    case xla::PrimitiveType::PRED: {
      zero.Set<bool>({}, false);
      break;
    }
    case PrimitiveType::S8: {
      zero.Set<int8_t>({}, 0);
      break;
    }
    case PrimitiveType::S16: {
      zero.Set<int16_t>({}, 0);
      break;
    }
    case xla::PrimitiveType::S32: {
      zero.Set<int32_t>({}, 0);
      break;
    }
    case xla::PrimitiveType::S64: {
      zero.Set<int64_t>({}, 0);
      break;
    }
    case xla::PrimitiveType::U8: {
      zero.Set<uint8_t>({}, 0);
      break;
    }
    case xla::PrimitiveType::F32: {
      zero.Set<float>({}, 0);
      break;
    }
    case xla::PrimitiveType::F64: {
      zero.Set<double>({}, 0);
      break;
    }
    default: {
      TF_LOG(FATAL) << "Not implemented yet: " << type;
    }
  }
  return ConstantLiteral(builder, zero);
}

XlaOp One(XlaBuilder* builder, PrimitiveType type) {
  Literal one(ShapeUtil::MakeShape(type, {}));
  switch (type) {
    case xla::PrimitiveType::PRED: {
      one.Set<bool>({}, true);
      break;
    }
    case xla::PrimitiveType::S8: {
      one.Set<int8_t>({}, 1);
      break;
    }
    case xla::PrimitiveType::S16: {
      one.Set<int16_t>({}, 1);
      break;
    }
    case PrimitiveType::S32: {
      one.Set<int32_t>({}, 1);
      break;
    }
    case PrimitiveType::S64: {
      one.Set<int64_t>({}, 1);
      break;
    }
    case xla::PrimitiveType::U8: {
      one.Set<uint8_t>({}, 1);
      break;
    }
    case xla::PrimitiveType::F32: {
      one.Set<float>({}, 1);
      break;
    }
    case xla::PrimitiveType::F64: {
      one.Set<double>({}, 1);
      break;
    }
    default: {
      TF_LOG(FATAL) << "Not implemented yet: " << type;
    }
  }
  return ConstantLiteral(builder, one);
}

}  // namespace xla
