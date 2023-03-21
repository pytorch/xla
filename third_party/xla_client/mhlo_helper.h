#ifndef XLA_MHLO_HELPER_H_
#define XLA_MHLO_HELPER_H_

#include "tensorflow/compiler/xla/client/xla_computation.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
}  // namespace mlir

namespace xla {

// void hlo_mhlo_hlo_roundtrip_helper(HloModuleProto* proto);

// void hlo_stablehlo_hlo_roundtrip_helper(HloModuleProto* proto);

void printHloModuleProto(const HloModuleProto* proto);

bool hlo_mhlo_helper(const HloModuleProto* proto, mlir::ModuleOp* mlir_module);

bool mhlo_hlo_helper(const mlir::ModuleOp* mlir_module, HloProto* proto);

bool mhlo_stablehlo_helper(mlir::ModuleOp* mlir_module, mlir::MLIRContext* context);

bool stablehlo_mhlo_helper(mlir::ModuleOp* mlir_module, mlir::MLIRContext* context);

}

#endif