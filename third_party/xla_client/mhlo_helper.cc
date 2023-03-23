#include "third_party/xla_client/mhlo_helper.h"
#include "third_party/xla_client/xla_util.h"
#include "tensorflow/compiler/xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "tensorflow/compiler/xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h" // from @llvm-project

#include <iostream>

namespace xla {

bool hlo_mhlo_helper(const HloModuleProto* proto, mlir::ModuleOp* mlir_module) {
  auto status = ConvertHloToMlirHlo(*mlir_module, proto, /*import_all_computations=*/false);
  if (!status.ok()) {
    std::cout << "hlo2mhlo not ok" << std::endl;
    return false;
  }
  if (!mlir::verify(*mlir_module).succeeded()) {
    std::cout << "mhlo from hlo2mhlo verify not ok" << std::endl;
    printHloModuleProto(proto);
    return false;
  }
  return true;
}

bool mhlo_hlo_helper(const mlir::ModuleOp* mlir_module, HloProto* hlo_proto) {
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = true;
  auto status1 = mlir::ConvertMlirHloToHlo(
    *mlir_module, hlo_proto, /*use_tuple_args=*/false, /*return_tuple=*/false,
    options);
  if (!status1.ok()) {
    std::cout << "mhlo2hlo not ok" << std::endl;
    return false;
  }
  return true;
}

bool mhlo_stablehlo_helper(mlir::ModuleOp* mlir_module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    std::cout << "mhlo to stablehlo not ok" << std::endl;
    std::cout << "stablehlo dump: " << std::endl;
    mlir_module->dump();
    return false;
  }
  return true;
}

bool stablehlo_mhlo_helper(mlir::ModuleOp* mlir_module, mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    std::cout << "mhlo to stablehlo not ok" << std::endl;
    std::cout << "mhlo dump: " << std::endl;
    mlir_module->dump();
    return false;
  }
  return true;
}

void printHloModuleProto(const HloModuleProto* proto) {
  auto hlo_module = xla::util::CreateModuleFromProto(*proto);
  std::cout << hlo_module.value()->ToString() << std::endl;
}

}