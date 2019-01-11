#pragma once

#include "ir.h"

namespace torch_xla {
namespace ir {
namespace ops {

struct XlaOpsArena {
  OpKind device_data;
};

const XlaOpsArena& XlaOps();

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
