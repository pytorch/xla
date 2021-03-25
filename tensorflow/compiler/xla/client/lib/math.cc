#include "tensorflow/compiler/xla/client/lib/math.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"

using namespace torch::jit::tensorexpr;

namespace xla {

XlaOp Erfc(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return erfc(x); }, "erfc");
}

XlaOp Erf(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return erf(x); }, "erf");
}

XlaOp RoundToEven(XlaOp x) {
  auto half = ScalarLike(x, 0.5);
  auto one = ScalarLike(x, 1.0);
  auto two = ScalarLike(x, 2.0);

  auto round_val = Floor(x);
  auto fraction = x - round_val;
  auto diff_val_nearest_even = round_val - two * Floor(half * x);
  auto is_odd = Eq(diff_val_nearest_even, one);
  return Select(Or(Gt(fraction, half), And(Eq(fraction, half), is_odd)),
                round_val + one, round_val);
}

XlaOp Acos(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return acos(x); }, "acos");
}

XlaOp Asin(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return asin(x); }, "asin");
}

XlaOp Atan(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return atan(x); }, "atan");
}

XlaOp Tan(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return tan(x); }, "tan");
}

XlaOp Cosh(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return cosh(x); }, "cosh");
}

XlaOp Sinh(XlaOp x) {
  return UnaryOp(x, [&](const ExprHandle& x) { return sinh(x); }, "sinh");
}

}  // namespace xla
