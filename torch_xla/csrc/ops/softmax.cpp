#include "torch_xla/csrc/ops/softmax.h"

#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/softmax_builder.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::XlaOp LowerSoftmax(xla::XlaOp input, xla::int64 dim,
                        const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildSoftmax(input, dim);
  return CastToScalarType(result, dtype);
}

xla::Shape NodeOutputShape(const Value& input,
                           const c10::optional<at::ScalarType>& dtype) {
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        input.shape(), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return input.shape();
}

}  // namespace

Softmax::Softmax(const Value& input, xla::int64 dim,
                 c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::softmax), {input},
           [&]() { return NodeOutputShape(input, dtype); },
           /*num_outputs=*/1,
           xla::util::MHash(dim, OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {}

NodePtr Softmax::Clone(OpList operands) const {
  return MakeNode<Softmax>(operands.at(0), dim_, dtype_);
}

XlaOpVector Softmax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerSoftmax(input, dim_, dtype_), loctx);
}

std::string Softmax::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
