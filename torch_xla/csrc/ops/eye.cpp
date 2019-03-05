#include "torch_xla/csrc/ops/eye.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {
namespace ops {

Eye::Eye(xla::int64 lines, xla::int64 cols, xla::PrimitiveType element_type)
    : Node(ir::OpKind(at::aten::eye), {},
           xla::ShapeUtil::MakeShape(element_type, {lines, cols}),
           /*num_outputs=*/1, xla::util::MHash(lines, cols, element_type)),
      lines_(lines),
      cols_(cols),
      element_type_(element_type) {}

XlaOpVector Eye::Lower(LoweringContext* loctx) const {
  xla::XlaOp output =
      xla::IdentityMatrix(loctx->builder(), element_type_, lines_, cols_);
  return ReturnOp(output, loctx);
}

std::string Eye::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", lines=" << lines_ << ", cols=" << cols_
     << ", element_type=" << element_type_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
