#include "torch_xla/csrc/ops/fill.h"

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/data_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const Value& input,
                           const std::vector<xla::int64>& size) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    

    for (int i = 0; i < size.size(); i++) {
        std::cout << size[i] << std::endl;
    }
    std::cout << "milad in fill::func " << input.shape() << std::endl;
    std::cout << xla::ShapeUtil::ElementsIn(input.shape()) << " " << input.shape().rank() << " " << GetShapeDimensionType(/*device=*/nullptr) << std::endl;
    xla::Shape tensor_shape = input.shape();
    std::cout << "ranks: " << tensor_shape.rank() << std::endl;
    xla::XlaOp op;
    for (int i = 0; i < tensor_shape.rank(); ++i) {
      if (tensor_shape.is_dynamic_dimension(i)) {
        std::cout << "Dynamic Dimension indx: " << i << std::endl;
        auto _size = xla::GetDimensionSize(operands[0], i);
        op = xla::SetDimensionSize(operands[0], _size, i);
      } else {
        std::cout << "Static Dimension: " << i << std::endl;
        op = operands[0];
      }
    }
    if (tensor_shape.rank() == 0) {
      std::cout << "zero rank" << std::endl;
      op = operands[0];
    }
    op = BuildExpand(op, size);

    return op;
  };
  std::cout << "NodeOutputShape " << input.shape() << std::endl;
  return InferOutputShape({input.shape()}, lower_for_shape_fn);
}

}  // namespace

Fill::Fill(const Value& input,
           const std::vector<xla::int64> size,
           const at::Scalar& value,
           const Device& device,
           const std::shared_ptr<xla::ComputationClient::Data> data,
           const xla::PrimitiveType type,
           const xla::Shape& shape)
    : Node(ir::OpKind(at::aten::fill), {input},
           [&]() { return NodeOutputShape(input, size); },
           /*num_outputs=*/1, xla::util::MHash(size)),
      size_(std::move(size)),
      value_(std::move(value)),
      device_(std::move(device)),
      data_(std::move(data)),
      type_(std::move(type)){}

NodePtr Fill::Clone(OpList operands) const {
  return MakeNode<Fill>(operands.at(0), size_, value_, device_, data_, type_, shape()); //TODO Milad: fix the shape() call
}

XlaOpVector Fill::Lower(LoweringContext* loctx) const {
  xla::Literal literal(xla::ShapeUtil::MakeShape(shape().element_type(), {}));
  literal.Set<xla::int32>({}, static_cast<xla::int32>(value_.toInt()));
  xla::XlaOp input = xla::ConstantLiteral(loctx->builder(), literal);
  if (shape().rank() > 0) {
    input = xla::Broadcast(input, shape().dimensions());
  }
  //xla::XlaOp op = loctx->GetOutputOp(operand(0));
  //return ReturnOp(op, loctx);
  std::cout << "milad in fill::lower " << shape()  << std::endl;
  xla::Shape tensor_shape = shape();
  std::cout << "ranks: " << tensor_shape.rank() << std::endl;
  for (int i = 0; i < tensor_shape.rank(); ++i) {
    if (tensor_shape.is_dynamic_dimension(i)) {
      std::cout << "Dynamic Dimension indx: " << i << std::endl;
      auto size = xla::GetDimensionSize(input, i);
      input = xla::SetDimensionSize(input, size, i);
    } else {
      std::cout << "Static Dimension indx: " << i << std::endl;
    }
  }

  return ReturnOp(BuildExpand(input, size_), loctx);

}

std::string Fill::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla






