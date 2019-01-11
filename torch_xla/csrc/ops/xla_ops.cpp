#include "ops/xla_ops.h"

namespace torch_xla {
namespace ir {
namespace ops {
namespace {

const XlaOpsArena* CreateOps() {
  XlaOpsArena* xops = new XlaOpsArena();
  xops->device_data = OpKind::Get("xla::device_data");
  return xops;
}

}  // namespace

const XlaOpsArena& XlaOps() {
  static const XlaOpsArena* xops = CreateOps();
  return *xops;
}

}  // namespace ops
}  // namespace ir
}  // namespace torch_xla
