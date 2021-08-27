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
                           const xla::Shape& _shape, const std::vector<xla::int64>& size) {
  auto lower_for_shape_fn =
      [&](absl::Span<const xla::XlaOp> operands) -> xla::XlaOp {
    std::cout << "[fill::NodeOutputShape] shape, dimension-size, rank: " << input.shape() << ", " << size.size() << input.shape().rank() << std::endl;
    std::cout << "[fill::NodeOutputShape] ElementsIn: " << xla::ShapeUtil::ElementsIn(input.shape()) << std::endl;

    xla::Shape tensor_shape = _shape;
    xla::XlaOp op = BuildExpand(operands[0], size);
    for (int i = 0; i < tensor_shape.rank(); ++i) {
      if (tensor_shape.is_dynamic_dimension(i)) {
        std::cout << "[fill::NodeOutputShape] Dynamic Dimension Indx: " << i << std::endl;
        auto _size = xla::GetDimensionSize(op, i);
        op = xla::SetDimensionSize(op, _size, i);
      } else {
        std::cout << "[fill::NodeOutputShape] Static Dimension Indx: " << i << std::endl;
      }
    }
    return op;
  };

  return InferOutputShape({_shape}, lower_for_shape_fn);
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
           [&]() { return NodeOutputShape(input, shape, size); },
           /*num_outputs=*/1, xla::util::MHash(size)),
      size_(std::move(size)),
      value_(std::move(value)),
      device_(std::move(device)),
      data_(std::move(data)),
      type_(std::move(type)){}

NodePtr Fill::Clone(OpList operands) const {
  return MakeNode<Fill>(operands.at(0), size_, value_, device_, data_, type_, shape());
}

XlaOpVector Fill::Lower(LoweringContext* loctx) const {
  xla::Literal literal(xla::ShapeUtil::MakeShape(shape().element_type(), {}));
  literal.Set<xla::int32>({}, static_cast<xla::int32>(value_.toInt()));
  xla::XlaOp op = xla::ConstantLiteral(loctx->builder(), literal);
  
  xla::Shape tensor_shape = shape();
  if (shape().rank() > 0) {
    op = xla::Broadcast(op, shape().dimensions());
  }
  std::cout << "[fill::Lower] shape, rank: " << shape() << tensor_shape.rank() << std::endl;
  if (!shape().dimensions().empty()) {
    op = BuildExpand(op, shape().dimensions());
  }
  
  for (int i = 0; i < tensor_shape.rank(); ++i) {
    if (tensor_shape.is_dynamic_dimension(i)) {
      std::cout << "[fill::Lower] Dynamic Dimension Indx: " << i << std::endl;
      auto size = xla::GetDimensionSize(loctx->GetOutputOp(operand(0)), i);
      op = xla::SetDimensionSize(op, size, i);
    } else {
      std::cout << "[fill::Lower] Static Dimension Indx: " << i << std::endl;
    }
  }
  return ReturnOp(op, loctx);
}

std::string Fill::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", size=(" << absl::StrJoin(size_, ", ") << ")";
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla






