#include "torch_xla/csrc/ops/diagonal.h"

#include <cmath>

#include "tensorflow/compiler/xla/shape_util.h"
#include "third_party/xla_client/debug_macros.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/matrix.h"

namespace torch_xla {

Diagonal::Diagonal(const torch::lazy::Value& input, int64_t offset,
                   int64_t dim1, int64_t dim2)
    : XlaNode(
          torch::lazy::OpKind(at::aten::diagonal), {input},
          [&]() {
            return MakeDiagonalShape(GetXlaShape(input), offset, dim1, dim2);
          },
          /*num_outputs=*/1, torch::lazy::MHash(offset, dim1, dim2)),
      offset_(offset),
      dim1_(dim1),
      dim2_(dim2) {}

torch::lazy::NodePtr Diagonal::Clone(torch::lazy::OpList operands) const {
  return torch::lazy::MakeNode<Diagonal>(operands.at(0), offset_, dim1_, dim2_);
}

XlaOpVector Diagonal::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  xla::XlaOp output = BuildDiagonal(input, offset_, dim1_, dim2_);
  return ReturnOp(output, loctx);
}

std::string Diagonal::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", offset=" << offset_ << ", dim1=" << dim1_
     << ", dim2=" << dim2_;
  return ss.str();
}

xla::Shape Diagonal::MakeDiagonalShape(const xla::Shape& shape, int64_t offset,
                                       int64_t dim1, int64_t dim2) {
  std::vector<int64_t> dimensions;
  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
    if (dim != dim1 && dim != dim2) {
      dimensions.push_back(shape.dimensions(dim));
    }
  }
  int64_t dsize;
  if (offset >= 0) {
    dsize = std::max<int64_t>(
        std::min(shape.dimensions(dim1), shape.dimensions(dim2) - offset), 0);
  } else {
    dsize = std::max<int64_t>(
        std::min(shape.dimensions(dim1) + offset, shape.dimensions(dim2)), 0);
  }
  dimensions.push_back(dsize);
  return xla::ShapeUtil::MakeShape(shape.element_type(), dimensions);
}

}  // namespace torch_xla
