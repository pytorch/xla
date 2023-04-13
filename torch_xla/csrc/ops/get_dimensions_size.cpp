#include "torch_xla/csrc/ops/get_dimensions_size.h"

#include "absl/strings/str_join.h"
#include "xla/shape_util.h"
#include "torch_xla/csrc/helpers.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"

namespace torch_xla {

GetDimensionsSize::GetDimensionsSize(const torch::lazy::Value& input,
                                     std::vector<int64_t> dimensions)
    : XlaNode(xla_get_dimensions_size, {input},
              xla::ShapeUtil::MakeShape(
                  GetShapeDimensionType(/*device=*/nullptr), {}),
              /*num_outputs=*/1, torch::lazy::MHash(dimensions)),
      dimensions_(std::move(dimensions)) {}

torch::lazy::NodePtr GetDimensionsSize::Clone(
    torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<GetDimensionsSize>(operands.at(0), dimensions_);
}

XlaOpVector GetDimensionsSize::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = XlaHelpers::GetDimensionsSize({input}, dimensions_).size;
  return ReturnOp(output, loctx);
}

std::string GetDimensionsSize::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dimensions=("
     << absl::StrJoin(dimensions_, ", ") << ")";
  return ss.str();
}

}  // namespace torch_xla
