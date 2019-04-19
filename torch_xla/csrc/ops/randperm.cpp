#include "torch_xla/csrc/ops/randperm.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

Randperm::Randperm(xla::int64 upper_bound, xla::PrimitiveType element_type)
    : Node(ir::OpKind(at::aten::randperm), {},
           xla::ShapeUtil::MakeShape(element_type, {upper_bound}),
           /*num_outputs=*/1,
           xla::util::MHash(upper_bound, static_cast<int>(element_type))),
      upper_bound_(upper_bound),
      element_type_(element_type) {}

XlaOpVector Randperm::Lower(LoweringContext* loctx) const {
  return ReturnOp(BuildRandperm(upper_bound_, element_type_, loctx->builder()),
                  loctx);
}

std::string Randperm::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", upper_bound=" << upper_bound_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
