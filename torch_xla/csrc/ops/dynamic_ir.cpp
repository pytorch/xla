#include "torch_xla/csrc/ops/dynamic_ir.h"

#include "absl/strings/str_join.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {

DimensionNode::DimensionNode(torch::lazy::OpKind op, OpList operands, torch::lazy::hash_t hash_seed):
  XlaNode(op, operands, xla::ShapeUtil::MakeShape(xla::S32, {}), /*num_outputs=*/1, 
  /* hash_seed */ hash_seed){}

std::string DimensionNode::ToString() const {
  return "DimensionNode";
}

SizeNode::SizeNode(XlaValue input, size_t dim):
    DimensionNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::size")}, {input}, torch::lazy::MHash(dim)),
    dim_(dim) {};

XlaOpVector SizeNode::Lower(LoweringContext* loctx) const {
  auto input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::GetDimensionSize(input, this->dim_), loctx);
}

int64_t SizeNode:: getStaticValue() const {
    return dynamic_cast<const XlaNode*>(operand(0).node)->shape(0).size(dim_);
}

std::string SizeNode::ToString() const {
  return "SizeNode";
}

SizeAdd::SizeAdd(XlaValue a, XlaValue b):
  DimensionNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::add")}, {a, b}, torch::lazy::MHash(1)) {};

int64_t SizeAdd::getStaticValue() const {
    return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() + dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeAdd::ToString() const {
  return "SizeAdd";
}

SizeMul::SizeMul(XlaValue a, XlaValue b):
  DimensionNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::mul")}, {a, b}, torch::lazy::MHash(1)) {};

int64_t SizeMul::getStaticValue() const {
    return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() * dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeMul::ToString() const {
  return "SizeMul";
}

SizeDiv::SizeDiv(XlaValue a, XlaValue b):
  DimensionNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::div")}, {a, b}, torch::lazy::MHash(1)) {};

int64_t SizeDiv::getStaticValue() const {
    TORCH_CHECK(dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue() != 0, "Can't divide a dimension by zero");
    return dynamic_cast<const DimensionNode*>(operand(0).node)->getStaticValue() / dynamic_cast<const DimensionNode*>(operand(1).node)->getStaticValue();
}

std::string SizeDiv::ToString() const {
  return "SizeDiv";
}

}  // namespace torch_xla