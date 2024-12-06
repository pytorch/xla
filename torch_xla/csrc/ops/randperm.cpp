#include "torch_xla/csrc/ops/randperm.h"

#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/infer_output_shape.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "tsl/platform/stacktrace.h"
#include "tsl/platform/statusor.h"
#include "xla/hlo/builder/lib/loops.h"
#include "xla/shape_util.h"

namespace torch_xla {
namespace {

using namespace xla;

xla::Shape NodeOutputShape(int64_t n) {
  return xla::ShapeUtil::MakeShape(xla::PrimitiveType::S64, {n});
}

XlaOp Swap(XlaOp input, XlaOp i, XlaOp j) {
  XlaOp i_value = xla::DynamicSlice(input, {i}, /*slice_sizes=*/{1});
  XlaOp j_value = xla::DynamicSlice(input, {j}, /*slice_sizes=*/{1});

  XlaOp write_i = xla::DynamicUpdateSlice(input, j_value, {i});
  XlaOp write_j = xla::DynamicUpdateSlice(write_i, i_value, {j});

  return write_j;
}

absl::StatusOr<std::vector<XlaOp>> LoopBodyFn(XlaOp i,
                                              absl::Span<const XlaOp> values,
                                              XlaBuilder* builder) {
  XlaOp input_array = values[0];
  XlaOp upper_bound_exclusive = values[1];

  XlaOp target_index = xla::RngUniform(
      i, upper_bound_exclusive,
      ShapeUtil::MakeShape(xla::PrimitiveType::S64, /*dimensions=*/{1}));

  XlaOp swapped_array = Swap(input_array, i, target_index);
  return std::vector<XlaOp>{swapped_array, upper_bound_exclusive};
}

}  // namespace

RandPerm::RandPerm(int64_t n, const at::ScalarType dtype,
                   const at::Layout layout, const at::Device device,
                   bool pin_memory)
    : XlaNode(
          torch::lazy::OpKind(at::aten::randperm), /*operands=*/{},
          [&]() { return NodeOutputShape(n); }, /*num_outputs=*/1,
          torch::lazy::MHash(n)),
      n_(n) {}

// Fischer Yates Shuffle.
XlaOpVector RandPerm::Lower(LoweringContext* lotcx) const {
  xla::XlaBuilder* builder = lotcx->builder();
  auto init_tensor = xla::Iota(lotcx->builder(), xla::PrimitiveType::S64, n_);

  auto upper_bound_exclusive = xla::ConstantLiteral(
      lotcx->builder(), xla::LiteralUtil::CreateR0<int64_t>(n_));
  auto fischer_yates_loop = xla::ForEachIndex(
      /*num_iterations=*/n_ - 1, xla::PrimitiveType::S64, &LoopBodyFn,
      {init_tensor, upper_bound_exclusive}, "Fischer-Yates-Shuffle", builder);

  return ReturnOp(fischer_yates_loop.value()[0], lotcx);
}

std::string RandPerm::ToString() const {
  std::stringstream ss;
  ss << XlaNode::ToString() << ", n=" << n_;
  return ss.str();
}

}  // namespace torch_xla
