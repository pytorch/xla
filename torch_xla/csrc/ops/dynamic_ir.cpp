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
              {input}, xla::ShapeUtil::MakeShape(xla::S32, {}), 1,
              torch::lazy::MHash(dim)),
      dim_(dim) {
  // if (input.node->shapes().size() > 0) {
  //   for (int i = 0; i < operands().size(); i++) {
  //     std::cout << "operands i (is_symbolic): "
  //               << operands()[0].node->shape(0).is_symbolic().value()[i]
  //               << std::endl;
  //     // std::cout << "" << << std::endl;
  //   }
  // }
};

XlaOpVector SizeNode::Lower(LoweringContext* loctx) const {
  auto input = loctx->GetOutputOp(operand(0));
  return ReturnOp(xla::GetDimensionSize(input, this->dim_), loctx);
}

int64_t SizeNode::getStaticValue() const {
  return operand(0).node->shape(0).size(dim_);
}

bool SizeNode::isSymbolic() const {
  auto symbolic_vec = operand(0).node->shape(0).is_symbolic();
  if (!symbolic_vec.has_value()) {
    return true;
  }
  return symbolic_vec->at(dim_);
}

std::string SizeNode::ToString() const { return "SizeNode"; }

SizeAdd::SizeAdd(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::add")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S32, {}), 1){};

int64_t SizeAdd::getStaticValue() const {
  return dynamic_cast<const torch::lazy::DimensionNode*>(operand(0).node)
             ->getStaticValue() +
         dynamic_cast<const torch::lazy::DimensionNode*>(operand(1).node)
             ->getStaticValue();
}

bool SizeAdd::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeAdd::ToString() const { return "SizeAdd"; }

XlaOpVector SizeAdd::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(
      (xla::GetDimensionSize(input1, 0) + xla::GetDimensionSize(input2, 0)),
      loctx);
}

SizeMul::SizeMul(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::mul")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S32, {}), 1){};

int64_t SizeMul::getStaticValue() const {
  return dynamic_cast<const torch::lazy::DimensionNode*>(operand(0).node)
             ->getStaticValue() *
         dynamic_cast<const torch::lazy::DimensionNode*>(operand(1).node)
             ->getStaticValue();
}

bool SizeMul::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeMul::ToString() const { return "SizeMul"; }

XlaOpVector SizeMul::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::Mul(xla::GetDimensionSize(input1, 0),
                           xla::GetDimensionSize(input2, 0)),
                  loctx);
}

SizeDiv::SizeDiv(torch::lazy::Value a, torch::lazy::Value b)
    : XlaNode(torch::lazy::OpKind{c10::Symbol::fromQualString("aten::div")},
              {a, b}, xla::ShapeUtil::MakeShape(xla::S32, {}), 1){};

int64_t SizeDiv::getStaticValue() const {
  XLA_CHECK(dynamic_cast<const torch::lazy::DimensionNode*>(operand(1).node)
                ->getStaticValue() != 0)
      << "Can't divide a dimension by zero";
  return dynamic_cast<const torch::lazy::DimensionNode*>(operand(0).node)
             ->getStaticValue() /
         dynamic_cast<const torch::lazy::DimensionNode*>(operand(1).node)
             ->getStaticValue();
}

bool SizeDiv::isSymbolic() const {
  return DimCast(operand(0))->isSymbolic() || DimCast(operand(1))->isSymbolic();
}

std::string SizeDiv::ToString() const { return "SizeDiv"; }

XlaOpVector SizeDiv::Lower(LoweringContext* loctx) const {
  auto input1 = loctx->GetOutputOp(operand(0));
  auto input2 = loctx->GetOutputOp(operand(1));
  return ReturnOp(xla::Div(xla::GetDimensionSize(input1, 0),
                           xla::GetDimensionSize(input2, 0)),
                  loctx);
}

}  // namespace torch_xla
