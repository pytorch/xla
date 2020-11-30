#include "torch_xla/csrc/ops/amp_update_scale.h"

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_client/util.h"
#include "torch_xla/csrc/lowering_context.h"
#include "torch_xla/csrc/ops/xla_ops.h"
#include "torch_xla/csrc/tensor_util.h"
#include "torch_xla/csrc/xla_lower_util.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

xla::Shape NodeOutputShape(const OpList& inputs) {
  std::vector<xla::Shape> output_shapes;
  for (size_t i = 0; i < 2; ++i) {
    const xla::Shape& input_shape = inputs[i].shape();
    output_shapes.push_back(input_shape);
  }
  return xla::ShapeUtil::MakeTupleShape(output_shapes);
}

}  // namespace

AmpUpdateScale::AmpUpdateScale(const OpList& inputs)
    : Node(xla_amp_update_scale, inputs, NodeOutputShape(inputs),
           /*num_outputs=*/2) {}

NodePtr AmpUpdateScale::Clone(OpList operands) const {
  return MakeNode<AmpUpdateScale>(operands);
}

XlaOpVector AmpUpdateScale::Lower(LoweringContext* loctx) const {
  std::vector<xla::XlaOp> inputs;
  for (size_t i = 0; i < 6; ++i) {
    inputs.push_back(loctx->GetOutputOp(operand(i)));
  }
  return ReturnOps(BuildAmpUpdateScale(inputs), loctx);
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla