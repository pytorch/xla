#include "torch_xla/csrc/ops/get_dimensions_size.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {
namespace ir {
namespace ops {

GetDimensionsSize::GetDimensionsSize(const Value& input,
                                     std::vector<xla::int64> dimensions)
    : Node(xla_get_dimensions_size, {input},
           xla::ShapeUtil::MakeShape(GetShapeDimensionType(/*device=*/nullptr),
                                     {}),
           /*num_outputs=*/1, xla::util::MHash(dimensions)),
      dimensions_(std::move(dimensions)) {}

NodePtr GetDimensionsSize::Clone(OpList operands) const {
  return MakeNode<GetDimensionsSize>(operands.at(0), dimensions_);
}

XlaOpVector GetDimensionsSize::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = XlaHelpers::GetDimensionsSize({input}, dimensions_).size;
  return ReturnOp(output, loctx);
}

std::string GetDimensionsSize::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dimensions=(" << absl::StrJoin(dimensions_, ", ")
     << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
