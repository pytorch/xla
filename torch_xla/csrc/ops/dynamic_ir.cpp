#include "torch_xla/csrc/ops/dynamic_ir.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/tensor.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_graph_executor.h"

namespace torch_xla {

const torch::lazy::DimensionNode* DimCast(const torch::lazy::Node* node) {
  return dynamic_cast<const torch::lazy::DimensionNode*>(node);
}

const torch::lazy::DimensionNode* DimCast(torch::lazy::Output output) {
  return DimCast(output.node);
}

const std::shared_ptr<torch::lazy::DimensionNode> DimCast(
    const torch::lazy::NodePtr& node) {
  return std::dynamic_pointer_cast<torch::lazy::DimensionNode>(node);
}

SizeNode::SizeNode(torch::lazy::Value input, size_t dim)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::size")},
              {input},
              xla::ShapeUtil::MakeShape(
                  GetShapeDimensionType(/*device=*/nullptr), {}),
              1, torch::lazy::MHash(dim)),
      dim_(dim) {
  // Not all IR has torch::lazy::shape now, use xla::shape to unblock
  // the development.
  const XlaNode* xla_node = dynamic_cast<const XlaNode*>(operand(0).node);
  // We don't need to hash upper_bound_  because it is computed
  // from input shapes and input Node already hash its shape.
  upper_bound_ = xla_node->xla_shape(operand(0).index).dimensions(dim_);
};

int64_t SizeNode::getDynamicValue() const {
  if (dynamic_value_computed_) {
    TORCH_LAZY_COUNTER("CachedSizeNodeValue", 1);
    return runtime_size_;
  }
  torch::lazy::NodePtr cloned =
      torch::lazy::MakeNode<SizeNode>(operands_[0], dim_);
  // Wrap the IR of SizeNode into a dummy tensor and execute/fetch the value
  // of this tensor. GetTensors will return a cpu at::Tensor so we can just
  // extract the value of it.
  std::vector<XLATensorPtr> dummy_size_tensors = {
      XLATensor::Create(cloned, *GetDefaultDevice(), at::ScalarType::Long)};
  std::vector<at::Tensor> res =
      XLAGraphExecutor::Get()->GetTensors(&dummy_size_tensors);
  runtime_size_ = res[0].item().toInt();
  dynamic_value_computed_ = true;
  return runtime_size_;
}

XlaOpVector SizeNode::Lower(LoweringContext* loctx) const {
  auto input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::GetDimensionSize(input, this->dim_), loctx);
}

std::string SizeNode::ToString() const { return "aten::size for size"; }

SizeAdd::SizeAdd(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::add")},
              {a, b},
              xla::ShapeUtil::MakeShape(
                  GetShapeDimensionType(/*device=*/nullptr), {}),
              1) {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  // SizeAdd can only be perfomed between two DimensionNode
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  // We don't need to hash upper_bound_ and because it is computed
  // from input shapes and input Node already hash its shape.
  upper_bound_ = dim_node_0->getStaticValue() + dim_node_1->getStaticValue();
};

int64_t SizeAdd::getDynamicValue() const {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  return dim_node_0->getDynamicValue() + dim_node_1->getDynamicValue();
}

std::string SizeAdd::ToString() const { return "aten::add for size"; }

XlaOpVector SizeAdd::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp((input1 + input2), loctx);
}

SizeEq::SizeEq(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::eq")},
              {a, b}, GetShapeDimensionType(/*device=*/nullptr), {}), 1) {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
};

int64_t SizeEq::getDynamicValue() const {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  return dim_node_0->getDynamicValue() == dim_node_1->getDynamicValue() ? 1 : 0;
}

std::string SizeEq::ToString() const { return "SizeEq"; }

SizeConstant::SizeConstant(int64_t val) : Scalar(c10::Scalar{val}, xla::S64){};

SizeMul::SizeMul(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::mul")},
              {a, b}, GetShapeDimensionType(/*device=*/nullptr), {}), 1) {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  // SizeMul can only be perfomed between two DimensionNode
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  // We don't need to hash upper_bound_ and because it is computed
  // from input shapes and input Node already hash its shape.
  upper_bound_ = dim_node_0->getStaticValue() * dim_node_1->getStaticValue();
};

int64_t SizeMul::getDynamicValue() const {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  return dim_node_0->getDynamicValue() * dim_node_1->getDynamicValue();
}

std::string SizeMul::ToString() const { return "SizeMul"; }

XlaOpVector SizeMul::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(input1 * input2, loctx);
}

SizeDiv::SizeDiv(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::div")},
              {a, b}, GetShapeDimensionType(/*device=*/nullptr), {}), 1) {
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

int64_t SizeDiv::getDynamicValue() const {
  const torch::lazy::DimensionNode* dim_node_0 = DimCast(operand(0));
  const torch::lazy::DimensionNode* dim_node_1 = DimCast(operand(1));
  XLA_CHECK(dim_node_0);
  XLA_CHECK(dim_node_1);
  XLA_CHECK(dim_node_1->getDynamicValue() != 0)
      << "Can't divide a dynamic dimension by zero";
  return dim_node_0->getDynamicValue() / dim_node_1->getDynamicValue();
}

std::string SizeDiv::ToString() const { return "SizeDiv"; }

XlaOpVector SizeDiv::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::Div(input1, input2), loctx);
}

}  // namespace torch_xla
