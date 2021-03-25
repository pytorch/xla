#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace xla {

inline XlaOp Square(XlaOp operand) { TF_LOG(FATAL) << "Not implemented yet."; }

inline XlaOp Reciprocal(XlaOp operand) {
  TF_LOG(FATAL) << "Not implemented yet.";
}

XlaOp Erfc(XlaOp x);

XlaOp Erf(XlaOp x);

inline XlaOp ErfInv(XlaOp x) { TF_LOG(FATAL) << "Not implemented yet."; }

XlaOp RoundToEven(XlaOp x);

XlaOp Acos(XlaOp x);

XlaOp Asin(XlaOp x);

XlaOp Atan(XlaOp x);

XlaOp Tan(XlaOp x);

inline XlaOp Acosh(XlaOp x) { TF_LOG(FATAL) << "Not implemented yet."; }

inline XlaOp Asinh(XlaOp x) { TF_LOG(FATAL) << "Not implemented yet."; }

inline XlaOp Atanh(XlaOp x) { TF_LOG(FATAL) << "Not implemented yet."; }

XlaOp Cosh(XlaOp x);

XlaOp Sinh(XlaOp x);

}  // namespace xla
