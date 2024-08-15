#include "torch_xla/csrc/ops/dot_general.h"

#include <c10/core/ScalarType.h>
#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/dtype.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"

namespace torch_xla {

namespace {

xla::XlaOp BuildDotGeneral(
    const xla::XlaOp& lhs, const xla::XlaOp& rhs,
    const std::vector<std::vector<int>>& dim_vectors,
    std::optional<at::ScalarType> preferred_element_type) {
  xla::DotDimensionNumbers dot_dim_numbers;
  dot_dim_numbers.mutable_lhs_contracting_dimensions()->Add(
      dim_vectors[0].begin(), dim_vectors[0].end());
  dot_dim_numbers.mutable_rhs_contracting_dimensions()->Add(
      dim_vectors[1].begin(), dim_vectors[1].end());
  dot_dim_numbers.mutable_lhs_batch_dimensions()->Add(dim_vectors[2].begin(),
                                                      dim_vectors[2].end());
  dot_dim_numbers.mutable_rhs_batch_dimensions()->Add(dim_vectors[3].begin(),
                                                      dim_vectors[3].end());
  std::optional<xla::PrimitiveType> xla_preferred_element_type;
  if (preferred_element_type.has_value()) {
    xla_preferred_element_type =
        XlaTypeFromTorchType(preferred_element_type.value());
  }
  return xla::DotGeneral(lhs, rhs, dot_dim_numbers,
                         /*precision_config=*/nullptr,
                         /*preferred_element_type=*/xla_preferred_element_type);
}

xla::Shape NodeOutputShape(
    const torch::lazy::Value& lhs, const torch::lazy::Value& rhs,
    const std::vector<std::vector<int>>& dim_vectors,
    std::optional<at::ScalarType> preferred_element_type) {
  auto lower_for_shape_fn =
      [dim_vectors, preferred_element_type](
          absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    XLA_CHECK(operands.size() == 2);
    return BuildDotGeneral(operands[0], operands[1], dim_vectors,
                           preferred_element_type);
  };
  return InferOutputShape({GetXlaShape(lhs), GetXlaShape(rhs)},
                          lower_for_shape_fn);
}

}  // namespace

DotGeneral::DotGeneral(const torch::lazy::Value& lhs,
                       const torch::lazy::Value& rhs,
                       const std::vector<std::vector<int>>& dim_vectors,
                       std::optional<at::ScalarType> preferred_element_type)
    : XlaNode(
          xla_dot_general, {lhs, rhs},
          [&]() {
            return NodeOutputShape(lhs, rhs, dim_vectors,
                                   preferred_element_type);
          },
          /*num_outputs=*/1,
          torch::lazy::MHash(dim_vectors, preferred_element_type)),
      dim_vectors_(dim_vectors),
      preferred_element_type_(preferred_element_type) {}

torch::lazy::NodePtr DotGeneral::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<DotGeneral>(
      operands.at(0), operands.at(1), dim_vectors_, preferred_element_type_);
}

XlaOpVector DotGeneral::Lower(LoweringContext* loctx) const {
  xla::XlaOp lhs = loctx->GetOutputOp(operand(0));
  xla::XlaOp rhs = loctx->GetOutputOp(operand(1));
  xla::XlaOp output =
      BuildDotGeneral(lhs, rhs, dim_vectors_, preferred_element_type_);
  return ReturnOp(output, loctx);
}

std::string DotGeneral::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", preferred_element_type=";
  return ss.str();
}

}  // namespace torch_xla
