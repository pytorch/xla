#include "torch_xla/csrc/ops/get_dimension_size.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {

GetDimensionSize::GetDimensionSize(const Value& input, xla::int64 dimension)
    : Node(xla_get_dimension_size, {input},
           xla::ShapeUtil::MakeShape(xla::PrimitiveType::S32, {}),
           /*num_outputs=*/1, xla::util::MHash(dimension)),
      dimension_(dimension) {}

NodePtr GetDimensionSize::Clone(OpList operands) const {
  return MakeNode<GetDimensionSize>(operands.at(0), dimension_);
}

XlaOpVector GetDimensionSize::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = xla::GetDimensionSize(input, dimension_);
  return ReturnOp(output, loctx);
}

std::string GetDimensionSize::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimension=" << dimension_;
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
