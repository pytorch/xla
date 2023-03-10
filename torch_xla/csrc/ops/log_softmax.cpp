#include "torch_xla/csrc/ops/log_softmax.h"

#include <torch/csrc/lazy/core/tensor_util.h>

#include "torch_xla/csrc/convert_ops.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/softmax_builder.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/torch_util.h"

namespace torch_xla {
namespace {

xla::XlaOp LowerLogSoftmax(xla::XlaOp input, int64_t dim,
                           const c10::optional<at::ScalarType>& dtype) {
  xla::XlaOp result = BuildLogSoftmax(input, dim);
  return CastToScalarType(result, dtype);
}

xla::Shape NodeOutputShape(const torch::lazy::Value& input,
                           const c10::optional<at::ScalarType>& dtype) {
  if (dtype) {
    return xla::ShapeUtil::ChangeElementType(
        GetXlaShape(input), MakeXlaPrimitiveType(*dtype, /*device=*/nullptr));
  }
  return GetXlaShape(input);
}

}  // namespace

LogSoftmax::LogSoftmax(const torch::lazy::Value& input, int64_t dim,
                       c10::optional<at::ScalarType> dtype,
                       std::vector<torch::lazy::Shape>&& shapes)
    : XlaNode(torch::lazy::OpKind(at::aten::log_softmax), {input},
              std::move(shapes),
              [&]() { return NodeOutputShape(input, dtype); },
              /*num_outputs=*/1,
              torch::lazy::MHash(dim, torch::lazy::OptionalOr<int>(dtype, -1))),
      dim_(dim),
      dtype_(dtype) {}

XlaOpVector LogSoftmax::Lower(LoweringContext* loctx) const {
  xla::XlaOp input = loctx->GetOutputOp(operand(0));
  return ReturnOp(LowerLogSoftmax(input, dim_, dtype_), loctx);
}

std::string LogSoftmax::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", dim=" << dim_
     << ", dtype=" << torch::lazy::OptionalOr<int>(dtype_, -1);
  return ss.str();
}

}  // namespace torch_xla
