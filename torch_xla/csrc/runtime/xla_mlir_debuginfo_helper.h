#ifndef XLA_MLIR_DEBUGINFO_HELPER_H_
#define XLA_MLIR_DEBUGINFO_HELPER_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace torch_xla {
namespace runtime {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreatePrepareXlaMlirDebuginfoPass();

}  // namespace runtime
}  // namespace torch_xla

#endif
