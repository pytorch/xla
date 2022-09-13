#include "torch_xla/csrc/ops/dynamic_ir.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

static const torch::lazy::DimensionNode* DimCast(torch::lazy::Output output) {
  return dynamic_cast<const torch::lazy::DimensionNode*>(output.node);
}

namespace torch_xla {

SizeNode::SizeNode(torch::lazy::Value input, size_t dim)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::size")},
              {input}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1,
              torch::lazy::MHash(dim)),
      dim_(dim){};

XlaOpVector SizeNode::Lower(LoweringContext* loctx) const {
  auto input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::GetDimensionSize(input, this->dim_), loctx);
}

int64_t SizeNode::getStaticValue() const {
  // Not all IR has torch::lazy::shape now, use xla::shape to unblock
  // the development.
  return dynamic_cast<const XlaNode*>(operand(0).node)
      ->xla_shape(operand(0).index)
      .dimensions(dim_);
}

std::string SizeNode::ToString() const { return "SizeNode"; }

SizeAdd::SizeAdd(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::add")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1) {
  // SizeAdd can only be perfomed between two DimensionNode
  XLA_CHECK(DimCast(operand(0)));
  XLA_CHECK(DimCast(operand(1)));
};

int64_t SizeAdd::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() +
         DimCast(operand(1))->getStaticValue();
}

std::string SizeAdd::ToString() const { return "SizeAdd"; }

XlaOpVector SizeAdd::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp((input1 + input2), loctx);
}

SizeMul::SizeMul(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::mul")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1) {
  // SizeMul can only be perfomed between two DimensionNode
  XLA_CHECK(DimCast(operand(0)));
  XLA_CHECK(DimCast(operand(1)));
};

int64_t SizeMul::getStaticValue() const {
  return DimCast(operand(0))->getStaticValue() *
         DimCast(operand(1))->getStaticValue();
}

std::string SizeMul::ToString() const { return "SizeMul"; }

XlaOpVector SizeMul::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(input1 * input2, loctx);
}

SizeDiv::SizeDiv(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::div")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S64, {}), 1) {
  // SizeDiv can only be perfomed between two DimensionNode
  XLA_CHECK(DimCast(operand(0)));
  XLA_CHECK(DimCast(operand(1)));
};

int64_t SizeDiv::getStaticValue() const {
  XLA_CHECK(DimCast(operand(1))->getStaticValue() != 0)
      << "Can't divide a dimension by zero";
  return DimCast(operand(0))->getStaticValue() /
         DimCast(operand(1))->getStaticValue();
}

std::string SizeDiv::ToString() const { return "SizeDiv"; }

XlaOpVector SizeDiv::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::Div(input1, input2), loctx);
}

}  // namespace torch_xla
