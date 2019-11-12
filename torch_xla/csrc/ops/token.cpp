#include "torch_xla/csrc/ops/token.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

Token::Token()
    : Node(xla_token, xla::ShapeUtil::MakeTokenShape(),
           /*num_outputs=*/1,
           /*hash_seed=*/0xe7396f95491f4493) {}

NodePtr Token::Clone(OpList operands) const { return MakeNode<Token>(); }

XlaOpVector Token::Lower(LoweringContext* loctx) const {
  return ReturnOp(xla::CreateToken(loctx->builder()), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
