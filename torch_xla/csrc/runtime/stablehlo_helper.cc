#include "torch_xla/csrc/runtime/stablehlo_helper.h"

#include <iostream>

#include "mlir/IR/Verifier.h"       // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"
#include "stablehlo/api/PortableApi.h"        // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"   // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"        // from @stablehlo
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"

namespace torch_xla {
namespace runtime {

static std::string getHloModuleStr(const xla::HloModuleProto* proto) {
  auto hlo_module = torch_xla::runtime::util::CreateModuleFromProto(*proto);
  return hlo_module.value()->ToString();
}

static std::string getMlirModuleStr(mlir::ModuleOp& mlir_module) {
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  // Enable Debug Info to include source line info in the StableHLO dump.
  mlir::OpPrintingFlags flags;
  static bool withSrcLineInfo =
      runtime::sys_util::GetEnvBool("XLA_HLO_DEBUG", false);
  if (withSrcLineInfo) {
    flags.enableDebugInfo(/*enable=*/true, /*prettyForm=*/true);
  }
  mlir_module.print(os, flags);
  return txt_mlir_module;
}

static std::string getMlirModuleBytecode(mlir::ModuleOp& mlir_module) {
  static bool from_pretty_print =
      runtime::sys_util::GetEnvBool("STABLEHLO_BYTECODE_FROM_PRETTYPRINT", false);
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  // TODO(lsiyuan): get the highest StableHLO version from runtime.
  const std::string stablehlo_version = "0.14.23";
  if (!from_pretty_print) {
    auto result = mlir::stablehlo::serializePortableArtifact(
        mlir_module, /* target_version = */ stablehlo_version, os);
    XLA_CHECK(result.succeeded()) << "Serializing StableHLO Failed";
  } else {
    std::string pretty_print_txt = getMlirModuleStr(mlir_module);
    auto result = mlir::stablehlo::serializePortableArtifact(
      pretty_print_txt, /* target_version = */ stablehlo_version, os);
    XLA_CHECK(result.succeeded()) << "Serializing StableHLO Failed";
  }
  return txt_mlir_module;
}

static absl::Status ConvertHloToMhlo(const xla::HloModuleProto* proto,
                                     mlir::ModuleOp* mlir_module) {
  auto status = xla::ConvertHloToMlirHlo(*mlir_module, proto,
                                         /*import_all_computations=*/false);
  if (!status.ok()) {
    return status;
  }
  if (!mlir::verify(*mlir_module).succeeded()) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "MHLO Module from HLO -> MHLO conversion is not legal.");
  }
  return absl::OkStatus();
}

static absl::Status mhloToStablehloHelper(mlir::ModuleOp* mlir_module,
                                          mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  // Apply pass to remove HLO tuple output, as MHLO/StableHLO supports multiple
  // outputs.
  pm.addPass(mlir::mhlo::createExpandHloTuplesPass());
  // Canonicalization after tuple flatten, to remove unused tuple op.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "StableHLO Module from MHLO -> StableHLO conversion is not leagal.");
  }
  return absl::OkStatus();
}

void ConvertHloToStableHlo(const xla::HloModuleProto* proto,
                           mlir::ModuleOp* mlir_module) {
  static const std::string err_msg =
      "Please open a github issue to PyTorch/XLA.\nOriginal HLO dump:\n";
  auto status = ConvertHloToMhlo(proto, mlir_module);
  XLA_CHECK(status.ok()) << "HLO -> MHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getHloModuleStr(proto);
  status = mhloToStablehloHelper(mlir_module, mlir_module->getContext());
  XLA_CHECK(status.ok()) << "MHLO -> StableHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getHloModuleStr(proto);
}

std::string hloToStablehlo(const xla::HloModuleProto* proto,
                           bool emit_bytecode) {
  mlir::MLIRContext context;
  mlir::ModuleOp mlir_module =
      mlir::ModuleOp::create(mlir::UnknownLoc::get(&context));
  ConvertHloToStableHlo(proto, &mlir_module);
  if (emit_bytecode) {
    return getMlirModuleBytecode(mlir_module);
  } else {
    return getMlirModuleStr(mlir_module);
  }
}

std::string GetHloModuleStr(const xla::HloModuleProto* proto) {
  auto hlo_module = runtime::util::CreateModuleFromProto(*proto);
  return hlo_module.value()->ToString();
}

static absl::Status ConvertStablehloToMhlo(mlir::ModuleOp* mlir_module,
                                           mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    return absl::Status(
        absl::StatusCode::kInternal,
        "StableHLO Module from StableHLO -> MHLO conversion is not leagal.");
  }
  return absl::OkStatus();
}

static absl::Status MhloToHloHelper(const mlir::ModuleOp* mlir_module,
                                    xla::HloProto* hlo_proto) {
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = true;
  auto status = mlir::ConvertMlirHloToHlo(*mlir_module, hlo_proto,
                                          /*use_tuple_args=*/false,
                                          /*return_tuple=*/false, options);
  if (!status.ok()) {
    return status;
  }
  return absl::OkStatus();
}

void ConvertStableHloToHlo(mlir::ModuleOp* mlir_module,
                           mlir::MLIRContext* context,
                           xla::HloProto* hlo_proto) {
  static const std::string err_msg =
      "Please open a github issue to PyTorch/XLA.\nOriginal StableHLO dump:\n";
  auto status = ConvertStablehloToMhlo(mlir_module, context);
  XLA_CHECK(status.ok()) << "StableHLO -> MHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getMlirModuleStr(*mlir_module);
  status = MhloToHloHelper(mlir_module, hlo_proto);
  XLA_CHECK(status.ok()) << "MHLO -> StableHLO conversion failed.\n"
                         << status.message() << err_msg
                         << getMlirModuleStr(*mlir_module);
}

}  // namespace runtime
}  // namespace torch_xla
