#ifndef STABLEHLO_HELPER_H_
#define STABLEHLO_HELPER_H_

#include "xla/client/xla_computation.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
}  // namespace mlir

namespace torch_xla {

std::string hloToStablehlo(const xla::HloModuleProto* proto,
                           bool emit_bytecode);

void ConvertHloToStableHlo(const xla::HloModuleProto* proto,
                           mlir::ModuleOp* mlir_module);

void ConvertStableHloToHlo(mlir::ModuleOp* mlir_module,
                           mlir::MLIRContext* context,
                           xla::HloProto* hlo_proto);

std::string GetHloModuleStr(const xla::HloModuleProto* proto);

const std::unordered_map<std::string, std::string>&
GetTorchDtypeToStablehloDtypeMap();

const std::unordered_map<xla::PrimitiveType, std::string>&
GetHloDtypeToStablehloDtypeMap();

}  // namespace torch_xla

#endif
