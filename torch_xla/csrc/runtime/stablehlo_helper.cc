#include "torch_xla/csrc/runtime/stablehlo_helper.h"

#include <iostream>

#include "mlir/IR/Verifier.h"       // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"
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

static std::string getMlirModuleBytecode(const mlir::ModuleOp& mlir_module) {
  std::string txt_mlir_module;
  llvm::raw_string_ostream os{txt_mlir_module};
  // TODO(lsiyuan): get the highest StableHLO version from runtime.
  auto result = mlir::stablehlo::serializePortableArtifact(
      mlir_module, /* target_version = */ "0.14.1", os);
  XLA_CHECK(result.succeeded()) << "Serializing StableHLO Failed";
  return txt_mlir_module;
}

static absl::Status hloToMhloHelper(const xla::HloModuleProto* proto,
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
  ;
}

void ConvertHloToStableHlo(const xla::HloModuleProto* proto,
                           mlir::ModuleOp* mlir_module) {
  static const std::string err_msg =
      "Please open a github issue to PyTorch/XLA.\nOriginal HLO dump:\n";
  auto status = hloToMhloHelper(proto, mlir_module);
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

void printHloModuleProto(const xla::HloModuleProto* proto) {
  auto hlo_module = runtime::util::CreateModuleFromProto(*proto);
  std::cout << "check hlo dump \n"
            << hlo_module.value()->ToString() << std::endl;
}

// stablehlo -> HLO
bool stablehlo_mhlo_helper(mlir::ModuleOp* mlir_module,
                           mlir::MLIRContext* context) {
  mlir::PassManager pm(context);
  pm.addPass(mlir::mhlo::createStablehloLegalizeToHloPass());
  if (!mlir::succeeded(pm.run(*mlir_module))) {
    std::cout << "mhlo to stablehlo not ok" << std::endl;
    // std::cout << "mhlo dump: " << std::endl;
    // mlir_module->dump();
    return false;
  }
  return true;
}

bool mhlo_hlo_helper(const mlir::ModuleOp* mlir_module,
                     xla::HloProto* hlo_proto) {
  mlir::MlirToHloConversionOptions options;
  options.propagate_layouts = true;
  auto status1 = mlir::ConvertMlirHloToHlo(*mlir_module, hlo_proto,
                                           /*use_tuple_args=*/false,
                                           /*return_tuple=*/false, options);
  if (!status1.ok()) {
    std::cout << "mhlo2hlo not ok" << std::endl;
    return false;
  }
  return true;
}

void convertStableHLOToHLO(mlir::ModuleOp* mlir_module,
                           mlir::MLIRContext* context,
                           xla::HloProto* hlo_proto) {
  // convert to hlo
  if (!stablehlo_mhlo_helper(mlir_module, context)) {
    std::cout << "stablehlo -> mhlo fail" << std::endl;
    return;
  }
  if (!mhlo_hlo_helper(mlir_module, hlo_proto)) {
    std::cout << "mhlo -> hlo fail" << std::endl;
    return;
  }
}

// Following is for stablehlo inference purpose.
static void loadSerializationDialects(mlir::MLIRContext* context) {
  context->loadDialect<mlir::func::FuncDialect>();
  context->loadDialect<mlir::stablehlo::StablehloDialect>();
  context->loadDialect<mlir::vhlo::VhloDialect>();
}

mlir::ModuleOp DeserializeStableHLO(const std::string& bytecode,
                                    mlir::MLIRContext* context) {
  loadSerializationDialects(context);
  auto module = mlir::stablehlo::deserializePortableArtifact(bytecode, context);
  XLA_CHECK(module) << "StableHLO deserialization failed.";
  mlir::ModuleOp mlir_module = *module;
  std::cout << "Deserialized MLIR Module: \n" << getMlirModuleStr(mlir_module);

  // test
  // convert to hlo
  if (!stablehlo_mhlo_helper(&mlir_module, context)) {
    return mlir_module;
  }
  xla::HloProto hlo_proto;
  if (!mhlo_hlo_helper(&mlir_module, &hlo_proto)) {
    return mlir_module;
  }

  // CompileInstance takes `XlaComputation`, which can be constructed from
  // `HloModuleProto`. `HloProto` contains `HloModuleProto`
  // TODO: figure out what else HloProto contains and if it's useful.
  xla::HloModuleProto* hlo_module_proto = hlo_proto.mutable_hlo_module();
  printHloModuleProto(hlo_module_proto);

  return mlir_module;
#if 0
  // convert to hlo
  if(!stablehlo_mhlo_helper(&mlir_module, &context)) {
    return mlir_module;
  }
  xla::HloProto hlo_proto;
  if (!mhlo_hlo_helper(&mlir_module, &hlo_proto)) {
    return;
  }

  // CompileInstance takes `XlaComputation`, which can be constructed from 
  // `HloModuleProto`. `HloProto` contains `HloModuleProto`
  // TODO: figure out what else HloProto contains and if it's useful.
  xla::HloModuleProto* hlo_module_proto = hlo_proto.mutable_hlo_module();
  printHloModuleProto(hlo_module_proto);
  xla::XlaComputation computation(*hlo_module_proto);

  xla::ProgramShape program_shape = ConsumeValue(computation.GetProgramShape());
  // output shape
  xla::Shape shape = MakeShapeWithDeviceLayout(
      program_shape.result(), static_cast<XlaDeviceType>(coll.device.type()));

  // Create PJRT computation client
  auto client = std::make_unique<PjRtComputationClient>();
  std::string device = client->GetDefaultDevice();
  std::vector<runtime::ComputationClient::CompileInstance> instances;
  instances.push_back({std::move(computation), device,
                       client->GetCompilationDevices(
                          device, client->GetLocalDevices()),
                       &shape});
#endif
}

}  // namespace runtime
}  // namespace torch_xla
