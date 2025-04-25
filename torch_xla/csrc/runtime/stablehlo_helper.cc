#include "torch_xla/csrc/runtime/stablehlo_helper.h"

#include <iostream>

#include "mlir/IR/Verifier.h"       // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"
#include "stablehlo/api/PortableApi.h"        // from @stablehlo
#include "stablehlo/dialect/Serialization.h"  // from @stablehlo
#include "stablehlo/dialect/StablehloOps.h"   // from @stablehlo
#include "stablehlo/dialect/Version.h"        // from @stablehlo
#include "stablehlo/dialect/VhloOps.h"        // from @stablehlo
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/stablehlo_composite_helper.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/xla_mlir_debuginfo_helper.h"
#include "torch_xla/csrc/runtime/xla_util.h"
#include "xla/hlo/translate/hlo_to_mhlo/hlo_to_mlir_hlo.h"
#include "xla/hlo/translate/mhlo_to_hlo/mlir_hlo_to_hlo.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

namespace torch_xla {

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
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  const std::string stablehlo_version =
      mlir::vhlo::Version::getCurrentVersion().toString();
  auto result = mlir::stablehlo::serializePortableArtifact(
      mlir_module, /* target_version = */ stablehlo_version, os);
  XLA_CHECK(result.succeeded()) << "Serializing StableHLO Failed";
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
  pm.addPass(torch_xla::runtime::CreatePrepareXlaMlirDebuginfoPass());
  // legalize `mhlo.dot` to `mhlo.dot_general` to workaround the shape
  // refinement issue in `stablehlo.dot`.
  // TODO(lsy323): Remove this pass when mhlo.dot will can be leagalized to
  // stablehlo.dot_general in MHLO->StableHLO converter. Or shape refinement
  // logic is fixed for stablehlo.dot.
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeDotToDotGeneralPass());
  // Apply pass to remove HLO tuple output, as MHLO/StableHLO supports multiple
  // outputs.
  pm.addPass(mlir::mhlo::createExpandHloTuplesPass());
  // Canonicalization after tuple flatten, to remove unused tuple op.
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
  // Group patterns into StableHLO composites.
  pm.addPass(torch_xla::runtime::CreateBuildStableHLOCompositePass());
  pm.addNestedPass<mlir::func::FuncOp>(
      torch_xla::runtime::CreateRemoveXlaMarkTensorOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
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

const std::string GetTorchDtypeToStablehloDtype(const std::string& dtype) {
  if (dtype == "torch.int8") return "i8";
  if (dtype == "torch.uint8") return "ui8";
  if (dtype == "torch.int16") return "i16";
  if (dtype == "torch.int32") return "i32";
  if (dtype == "torch.int64") return "i64";
  XLA_ERROR() << "Unsupported dtype for conversion to Stablehlo type: "
              << dtype;
}

const std::unordered_map<xla::PrimitiveType, std::string>&
GetHloDtypeToStablehloDtypeMap() {
  static const std::unordered_map<xla::PrimitiveType, std::string> m_{
      {xla::PrimitiveType::S4, "i4"},    {xla::PrimitiveType::S8, "i8"},
      {xla::PrimitiveType::S16, "i16"},  {xla::PrimitiveType::S32, "i32"},
      {xla::PrimitiveType::S64, "i64"},  {xla::PrimitiveType::U4, "ui4"},
      {xla::PrimitiveType::U8, "ui8"},   {xla::PrimitiveType::U16, "ui16"},
      {xla::PrimitiveType::U32, "ui32"}, {xla::PrimitiveType::U64, "ui64"},
      {xla::PrimitiveType::F16, "f16"},  {xla::PrimitiveType::BF16, "bf16"},
      {xla::PrimitiveType::F32, "f32"},  {xla::PrimitiveType::F64, "f64"},
  };
  return m_;
}

xla::PrimitiveType GetTorchIntDtypeToHloDtype(const std::string& dtype) {
  if (dtype == "torch.int8") return xla::PrimitiveType::S8;
  if (dtype == "torch.uint8") return xla::PrimitiveType::U8;
  if (dtype == "torch.int16") return xla::PrimitiveType::S16;
  if (dtype == "torch.int32") return xla::PrimitiveType::S32;
  if (dtype == "torch.int64") return xla::PrimitiveType::S64;
  XLA_ERROR() << "Unsupported dtype for conversion to Hlo type: " << dtype;
}

}  // namespace torch_xla
