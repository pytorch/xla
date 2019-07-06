#include "torch_xla/csrc/ops/log_softmax.h"

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

xla::XlaOp LowerLogSoftmax(const xla::XlaOp& input, xla::int64 dim,
                           const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildLogSoftmax(input, dim);
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

LogSoftmax::LogSoftmax(const Value& input, xla::int64 dim,
                       c10::optional<at::ScalarType> dtype)
    : Node(ir::OpKind(at::aten::log_softmax), {input},
           [&]() { return NodeOutputShape(input, dtype); },
           /*num_outputs=*/1,
           xla::util::MHash(dim, OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {}

NodePtr LogSoftmax::Clone(OpList operands) const {
  return MakeNode<LogSoftmax>(operands.at(0), dim_, dtype_);
}

XlaOpVector LogSoftmax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerLogSoftmax(input, dim_, dtype_), loctx);
}

std::string LogSoftmax::ToString() const {
  std::stringstream ss;
  ss << Node::ToString() << ", dim=" << dim_
     << ", dtype=" << OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
