#pragma once

#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "torch_xla/csrc/ir.h"
#include "torch_xla/csrc/lowering_context.h"

namespace torch_xla {
namespace ir {

class ShardingUtil {
 public:
  // Annotate HLO instructions in the lowered compuation by the embedded XLA
  // builder. For this call to be effective, this needs to be called after the
  // lowering and before building the computation; otherwise, this is a no-op.
  static void SetHloSharding(const ir::LoweringContext* lowering_ctx);
};

}  // namespace ir
}  // namespace torch_xla
