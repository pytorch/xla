#ifndef STABLEHLO_COMPOSITE_HELPER_H_
#define STABLEHLO_COMPOSITE_HELPER_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace torch_xla {
namespace runtime {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateBuildStableHLOCompositePass();

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateRemoveXlaMarkTensorOpsPass();

}  // namespace runtime
}  // namespace torch_xla

#endif
