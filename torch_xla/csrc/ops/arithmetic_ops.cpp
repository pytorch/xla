#include "ops/arithmetic_ops.h"
#include "helpers.h"
#include "lowering_context.h"
#include "ops/xla_ops.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"

namespace torch_xla {
namespace ir {
namespace ops {

namespace {

OpKind ArithmeticNodeKind(ArithmeticOp::Kind kind) {
  switch (kind) {
    case ArithmeticOp::Kind::Add: {
      return *xla_add;
    }
    case ArithmeticOp::Kind::Sub: {
      return *xla_sub;
    }
    case ArithmeticOp::Kind::Mul: {
      return *xla_mul;
    }
    case ArithmeticOp::Kind::Div: {
      return *xla_div;
    }
  }
}

}  // namespace

ArithmeticOp::ArithmeticOp(Kind kind, const NodeOperand& lhs,
                           const NodeOperand& rhs)
    : Node(ArithmeticNodeKind(kind), {lhs, rhs}), kind_(kind) {}

XlaOpVector ArithmeticOp::Lower(LoweringContext* loctx) const {
  const auto& lhs_rhs = operands();
  XLA_CHECK_EQ(lhs_rhs.size(), 2)
      << "Expected two operands, got " << lhs_rhs.size();
  auto lhs_op = loctx->GetOutputOp(lhs_rhs[0]);
  auto rhs_op = loctx->GetOutputOp(lhs_rhs[1]);
  switch (kind_) {
    case Kind::Add: {
      return {XlaHelpers::PromotedAdd(lhs_op, rhs_op)};
    }
    case Kind::Sub: {
      return {XlaHelpers::PromotedSub(lhs_op, rhs_op)};
    }
    case Kind::Mul: {
      return {XlaHelpers::PromotedMul(lhs_op, rhs_op)};
    }
    case Kind::Div: {
      return {XlaHelpers::PromotedDiv(lhs_op, rhs_op)};
    }
  }
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
