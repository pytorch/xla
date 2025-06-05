#include "torch_xla/csrc/runtime/xla_mlir_debuginfo_helper.h"

#include <cstring>
#include <string>
#include <utility>

#include "absl/log/log.h"

namespace torch_xla {
namespace runtime {

namespace {

// Defined in torch_xla/experimental/xla_mlir_debuginfo.py
static constexpr char XLA_MLIR_DEBUGINFO_BEGIN[] = "<XLA_MLIR_DEBUGINFO_BEGIN>";
static constexpr char XLA_MLIR_DEBUGINFO_END[] = "<XLA_MLIR_DEBUGINFO_END>";

class PrepareXlaMlirDebuginfoPass : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit PrepareXlaMlirDebuginfoPass()
      : mlir::OperationPass<mlir::ModuleOp>::OperationPass(
            mlir::TypeID::get<PrepareXlaMlirDebuginfoPass>()) {}

  ~PrepareXlaMlirDebuginfoPass() override = default;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    getOperation().walk([&](mlir::Operation* op) {
      llvm::SmallVector<std::string> debuginfos;
      ExtractXlaMlirDebuginfo(op->getLoc(), debuginfos);

      if (!debuginfos.empty()) {
        // If multiple debuginfos are found (which should be an exception),
        // pick arbitrary one and discard the rest;
        const std::string& debuginfo = debuginfos[0];
        op->setLoc(
            mlir::NameLoc::get(mlir::StringAttr::get(context, debuginfo)));
      }
      // TODO: Remove unspecified locations when a global flag is set.
    });
  }

  void ExtractXlaMlirDebuginfo(mlir::Location loc,
                               llvm::SmallVector<std::string>& debuginfos) {
    if (mlir::isa<mlir::FusedLoc>(loc)) {
      for (mlir::Location subloc :
           mlir::dyn_cast<mlir::FusedLoc>(loc).getLocations()) {
        ExtractXlaMlirDebuginfo(subloc, debuginfos);
      }
    }
    if (mlir::isa<mlir::NameLoc>(loc)) {
      std::string name(mlir::dyn_cast<mlir::NameLoc>(loc).getName().str());

      for (size_t i = 0; i < name.size();) {
        size_t begin = name.find(XLA_MLIR_DEBUGINFO_BEGIN, i);
        if (begin == std::string::npos) {
          break;
        }
        begin += strlen(XLA_MLIR_DEBUGINFO_BEGIN);
        size_t end = name.find(XLA_MLIR_DEBUGINFO_END, begin);
        if (end == std::string::npos) {
          break;
        }

        std::string debuginfo = name.substr(begin, end - begin);
        debuginfos.push_back(std::move(debuginfo));

        i = end + strlen(XLA_MLIR_DEBUGINFO_BEGIN);
      }
    }
    // TODO: Handle other loc types if debuginfo can be propagated/nested in
    // other loc type.
  }

  mlir::StringRef getName() const override {
    return llvm::getTypeName<PrepareXlaMlirDebuginfoPass>();
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<PrepareXlaMlirDebuginfoPass>(*this);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreatePrepareXlaMlirDebuginfoPass() {
  return std::make_unique<PrepareXlaMlirDebuginfoPass>();
}

}  // namespace runtime
}  // namespace torch_xla
