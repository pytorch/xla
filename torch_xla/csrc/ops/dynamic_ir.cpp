#include "torch_xla/csrc/ops/dynamic_ir.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/tensor.h"

static const torch::lazy::DimensionNode* DimCast(torch::lazy::Output output) {
  return dynamic_cast<const torch::lazy::DimensionNode*>(output.node);
}

namespace torch_xla {

SizeNode::SizeNode(torch::lazy::Value input, size_t dim)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::size")},
              {input}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1,
              torch::lazy::MHash(dim)),
      dim_(dim) {
  // Not all IR has torch::lazy::shape now, use xla::shape to unblock
  // the development.
  const XlaNode* xla_node = dynamic_cast<const XlaNode*>(operand(0).node);
  // We don't need to hash upper_bound_  because it is computed
  // from input shapes and input Node already hash its shape.
  upper_bound_ = xla_node->xla_shape(operand(0).index).dimensions(dim_);
};

int64_t SizeNode::getDynamicValue() const {
  torch::lazy::NodePtr cloned =
      torch::lazy::MakeNode<SizeNode>(operands_[0], dim_);
  std::vector<XLATensorPtr> dummy_size_tensors = {
      XLATensor::Create(cloned, *GetDefaultDevice(), at::ScalarType::Long)};
  // TODO: cache the result
  std::vector<at::Tensor> res = XLATensor::GetTensors(&dummy_size_tensors);
  return res[0].sum().item().toInt();
}

XlaOpVector SizeNode::Lower(LoweringContext* loctx) const {
  auto input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::GetDimensionSize(input, this->dim_), loctx);
}

std::string SizeNode::ToString() const { return "SizeNode"; }

SizeAdd::SizeAdd(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::add")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1) {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  // SizeAdd can only be perfomed between two DimensionNode
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  // We don't need to hash upper_bound_ and because it is computed
  // from input shapes and input Node already hash its shape.
  upper_bound_ = dim_node_0->getStaticValue() + dim_node_1->getStaticValue();
};

std::string SizeAdd::ToString() const { return "SizeAdd"; }

XlaOpVector SizeAdd::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp((input1 + input2), loctx);
}

SizeMul::SizeMul(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::mul")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1) {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  // SizeMul can only be perfomed between two DimensionNode
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  // We don't need to hash upper_bound_ and because it is computed
  // from input shapes and input Node already hash its shape.
  upper_bound_ = dim_node_0->getStaticValue() * dim_node_1->getStaticValue();
};

std::string SizeMul::ToString() const { return "SizeMul"; }

XlaOpVector SizeMul::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(input1 * input2, loctx);
}

SizeDiv::SizeDiv(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::div")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1) {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  // SizeDiv can only be perfomed between two DimensionNode
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  // We don't need to hash upper_bound_ and because it is computed
  // from input shapes and input Node already hash its shape.
  XLA_CHECK(dim_node_1->getStaticValue() != 0)
      << "Can't divide a dimension by zero";
  upper_bound_ = dim_node_0->getStaticValue() / dim_node_1->getStaticValue();
};

std::string SizeDiv::ToString() const { return "SizeDiv"; }

XlaOpVector SizeDiv::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::Div(input1, input2), loctx);
}

}  // namespace torch_xla
