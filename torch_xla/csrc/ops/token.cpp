#include "torch_xla/csrc/ops/token.h"

#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

// This should be using xla::ShapeUtil::MakeTokenShape() once we switch to the
// real Token based XLA AllReduce().
Token::Token()
    : Node(xla_token, xla::ShapeUtil::MakeShape(xla::PrimitiveType::F32, {}),
           /*num_outputs=*/1,
           /*hash_seed=*/0xe7396f95491f4493) {}

NodePtr Token::Clone(OpList operands) const { return MakeNode<Token>(); }

XlaOpVector Token::Lower(LoweringContext* loctx) const {
  // This should be using xla::CreateToken() once we have added Token support to
  // XLA AllReduce(). Meanwhile we use a constant as token, and we handle it
  // accordingly in cross_replica_reduces.cpp.
  return ReturnOp(xla::Zero(loctx->builder(), shape().element_type()), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
