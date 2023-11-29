#include "torch_xla/csrc/runtime/stablehlo_composite_helper.h"

#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mlir/IR/IRMapping.h"
#include "single_include/nlohmann/json.hpp"
#include "stablehlo/dialect/StablehloOps.h"

namespace torch_xla {
namespace runtime {

namespace {

using nlohmann::json;

static bool IsXlaMarkTensorOp(mlir::Operation* op) {
  if (op == nullptr) {
    return false;
  }
  if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
    return false;
  }
  if (!llvm::isa<mlir::stablehlo::CustomCallOp>(op)) {
    return false;
  }
  auto target_name =
      op->getAttr("call_target_name").dyn_cast<mlir::StringAttr>();
  if (target_name == nullptr || target_name.str() != "xla_mark_tensor") {
    return false;
  }
  return true;
}

struct BoundaryMetadata {
  std::string name;
  int64_t id;
  int64_t pos;
  bool is_input;
  std::unordered_map<std::string, json> attrs;

  std::string boundary_id() const { return absl::StrCat(name, "__", id); }

  auto uid() const { return std::forward_as_tuple(name, id, pos, is_input); }

  bool operator==(const BoundaryMetadata& other) const {
    return uid() == other.uid();
  }
  bool operator<(const BoundaryMetadata& other) const {
    return uid() < other.uid();
  }

  static std::unique_ptr<BoundaryMetadata> Parse(llvm::StringRef str) {
    auto j = json::parse(str, /*cb=*/nullptr, /*allow_exceptions=*/false);
    return Build(j);
  }

 private:
  template <typename T>
  static bool CopyJsonValue(const nlohmann::basic_json<>& j,
                            llvm::StringRef key, json::value_t expected_type,
                            T& to) {
    auto kv = j.find(key);

    if (kv == j.end()) {
      return false;
    }
    if (kv.value().type() != expected_type) {
      return false;
    }
    kv.value().get_to(to);
    return true;
  }

  static std::unique_ptr<BoundaryMetadata> Build(
      const nlohmann::basic_json<>& j) {
    BoundaryMetadata metadata;

    bool is_valid_metadata_json =
        CopyJsonValue(j, "name", json::value_t::string, metadata.name) &&
        CopyJsonValue(j, "id", json::value_t::number_unsigned, metadata.id) &&
        CopyJsonValue(j, "pos", json::value_t::number_unsigned, metadata.pos) &&
        CopyJsonValue(j, "is_input", json::value_t::boolean, metadata.is_input);

    if (!is_valid_metadata_json) {
      return nullptr;
    }

    if (auto kv = j.find("attr"); kv != j.end() && kv.value().is_object()) {
      auto& attrs_j = kv.value();
      for (auto attr_j = attrs_j.begin(); attr_j != attrs_j.end(); ++attr_j) {
        metadata.attrs.insert({attr_j.key(), attr_j.value()});
      }
    }
    return std::make_unique<BoundaryMetadata>(std::move(metadata));
  }
};

class BuildStableHLOCompositePass : public mlir::OperationPass<mlir::ModuleOp> {
 public:
  explicit BuildStableHLOCompositePass()
      : mlir::OperationPass<mlir::ModuleOp>::OperationPass(
            mlir::TypeID::get<BuildStableHLOCompositePass>()) {}

  ~BuildStableHLOCompositePass() override = default;

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    llvm::SmallVector<mlir::func::FuncOp> func_ops(
        module_op.getOps<mlir::func::FuncOp>());
    for (mlir::func::FuncOp& func_op : func_ops) {
      llvm::DenseMap<const mlir::Operation*, size_t> op_line_num =
          BuildOperationsLineNumberMap(func_op);
      for (auto op : func_op.getOps<mlir::stablehlo::CustomCallOp>()) {
        BuildStableHLOCompositeOp(op.getOperation(), op_line_num);
      }
    }
  }

  mlir::StringRef getName() const override {
    return llvm::getTypeName<BuildStableHLOCompositePass>();
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<BuildStableHLOCompositePass>(*this);
  }

 private:
  llvm::DenseMap<const mlir::Operation*, size_t> BuildOperationsLineNumberMap(
      mlir::func::FuncOp func_op) const {
    llvm::DenseMap<const mlir::Operation*, size_t> op_line_num;
    for (const auto& op : llvm::enumerate(func_op.getOps())) {
      op_line_num[&op.value()] = op.index();
    }
    return op_line_num;
  }

  std::unique_ptr<BoundaryMetadata> GetBoundaryMetadata(mlir::Operation* op) {
    if (!IsXlaMarkTensorOp(op)) {
      return nullptr;
    }
    auto backend_config =
        op->getAttr("backend_config").dyn_cast<mlir::StringAttr>();
    if (backend_config == nullptr) {
      return nullptr;
    }
    return BoundaryMetadata::Parse(backend_config);
  }

  mlir::DictionaryAttr BuildDictionaryAttrFromJsonMap(
      mlir::OpBuilder& builder,
      const std::unordered_map<std::string, json>& json_map) {
    llvm::SmallVector<mlir::NamedAttribute> named_attrs;
    for (auto& [key, j] : json_map) {
      switch (j.type()) {
        case json::value_t::number_integer:
        case json::value_t::number_unsigned:
          named_attrs.push_back(
              {builder.getStringAttr(key),
               builder.getI64IntegerAttr(j.template get<int64_t>())});
          break;
        case json::value_t::number_float:
          named_attrs.push_back(
              {builder.getStringAttr(key),
               builder.getF32FloatAttr(j.template get<float>())});
          break;
        case json::value_t::boolean:
          named_attrs.push_back({builder.getStringAttr(key),
                                 builder.getBoolAttr(j.template get<bool>())});
          break;
        case json::value_t::string:
          named_attrs.push_back(
              {builder.getStringAttr(key),
               builder.getStringAttr(j.template get<std::string>())});
          break;
        default:
          // Ignored unrecognizable attr json
          break;
      }
    }
    return builder.getDictionaryAttr(named_attrs);
  }

  void BuildStableHLOCompositeOp(
      mlir::Operation* op,
      const llvm::DenseMap<const mlir::Operation*, size_t>& op_line_num) {
    mlir::ModuleOp module_op = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::OpBuilder builder(context);

    std::unique_ptr<BoundaryMetadata> metadata = GetBoundaryMetadata(op);
    if (metadata == nullptr || metadata->is_input) {
      return;
    }
    const auto& output_metadata = *metadata;

    llvm::SetVector<mlir::Operation*> scope_ops_setvec;
    llvm::SetVector<std::pair<mlir::Value, int64_t>> arg_pos_setvec;
    llvm::SmallVector<mlir::Operation*> processing({op});

    // Reverse graph traversal: from boundary output op to boundary input op,
    // global function arg, or stablehlo constant.
    while (!processing.empty()) {
      mlir::Operation* curr_op = processing.back();
      processing.pop_back();
      if (scope_ops_setvec.contains(curr_op)) {
        continue;
      }

      if (auto curr_metadata_ptr = GetBoundaryMetadata(curr_op);
          curr_metadata_ptr != nullptr) {
        const auto& curr_metadata = *curr_metadata_ptr;
        if (curr_metadata.is_input &&
            curr_metadata.boundary_id() == output_metadata.boundary_id()) {
          // Terminal condition: boundary input op.
          arg_pos_setvec.insert({curr_op->getResult(0).dyn_cast<mlir::Value>(),
                                 curr_metadata.pos});
          continue;
        }
      }

      scope_ops_setvec.insert(curr_op);
      for (mlir::Value value : curr_op->getOperands()) {
        mlir::Operation* def_op = value.getDefiningOp();
        if (def_op == nullptr) {
          // Terminal condition: Global function arg
          arg_pos_setvec.insert({value, std::numeric_limits<int64_t>::max()});
        } else if (llvm::isa<mlir::stablehlo::ConstantOp>(op)) {
          // Terminal condition: constant
          scope_ops_setvec.insert(def_op);
        } else {
          processing.push_back(def_op);
        }
      }
    }
    // Sorts all ops within the boundary by their line numbers in the input
    // MLIR. The ops will be duplicated to the impl function following this
    // order.
    auto scope_ops = scope_ops_setvec.takeVector();
    for (auto& op : scope_ops) {
      if (!op_line_num.contains(op)) {
        return;
      }
    }
    std::sort(scope_ops.begin(), scope_ops.end(),
              [&op_line_num](const auto& a, const auto& b) {
                return op_line_num.at(a) < op_line_num.at(b);
              });

    // Sorts boundary args by their positions. Note that the args of the
    // composite and impl function may be more than the boundary inputs, because
    // the MLIR is lowered from the functionalized graph and additional args may
    // be Pytorch constants. In such case the position of those args would be
    // undetermined, while they would always come after boundary inputs.
    auto arg_pos_pairs = arg_pos_setvec.takeVector();
    std::stable_sort(
        arg_pos_pairs.begin(), arg_pos_pairs.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    llvm::SmallVector<mlir::Value> args;
    args.reserve(arg_pos_pairs.size());
    for (auto& [arg, unused] : arg_pos_pairs) {
      args.push_back(arg);
    }

    // Creates composite impl function and duplicates all ops within the
    // boundary in the function.
    llvm::SmallVector<mlir::Location> arg_locs;
    llvm::SmallVector<mlir::Type> arg_types,
        result_types(op->getResultTypes().begin(), op->getResultTypes().end());
    for (auto& arg : args) {
      arg_types.push_back(arg.getType());
      arg_locs.push_back(arg.getLoc());
    }

    mlir::func::FuncOp impl_func = builder.create<mlir::func::FuncOp>(
        module_op.getLoc(),
        absl::StrCat(output_metadata.boundary_id(), ".impl"),
        mlir::FunctionType::get(context, arg_types, result_types));
    mlir::IRMapping mapping;
    builder.createBlock(&impl_func.getBody(), impl_func.begin(), arg_types,
                        arg_locs);
    for (const auto& arg : llvm::enumerate(args)) {
      mapping.map(arg.value(), impl_func.getArgument(arg.index()));
    }
    for (mlir::Operation* original_op : scope_ops) {
      mlir::Operation* cloned_op = builder.clone(*original_op, mapping);
      mapping.map(original_op, cloned_op);
    }
    builder.create<mlir::func::ReturnOp>(impl_func.getBody().getLoc(),
                                         mapping.lookup(op)->getResults());

    // Adds the new function to symbol table.
    mlir::SymbolTable symbol_table(module_op);
    impl_func.setPrivate();
    symbol_table.insert(impl_func);

    builder.setInsertionPointAfter(op);
    llvm::SmallVector<mlir::NamedAttribute> call_attrs{
        {
            builder.getStringAttr("call_target_name"),
            builder.getStringAttr("stablehlo.composite"),
        },
        {
            builder.getStringAttr("called_computations"),
            builder.getArrayAttr(mlir::FlatSymbolRefAttr::get(
                builder.getContext(), impl_func.getSymName())),
        },
        {
            builder.getStringAttr("composite.backend_config"),
            builder.getDictionaryAttr(llvm::SmallVector<mlir::NamedAttribute>{
                {
                    builder.getStringAttr("attributes"),
                    BuildDictionaryAttrFromJsonMap(builder,
                                                   output_metadata.attrs),
                },
                {
                    builder.getStringAttr("name"),
                    builder.getStringAttr(output_metadata.name),
                },
            }),
        },
    };
    // Inserts composite call op.
    mlir::Operation* composite_call_op =
        builder.create<mlir::stablehlo::CustomCallOp>(
            op->getLoc(), impl_func.getFunctionType().getResults(), args,
            call_attrs);

    // Updates all users of this op's result(s) to use the results(s) of impl
    // func call.
    for (size_t i = 0; i < op->getNumResults(); ++i) {
      mlir::OpResult result = op->getResult(i);
      result.replaceAllUsesWith(composite_call_op->getResult(i));
    }

    // The unused scope_ops will be eliminated with canonicalizer.
  }
};

class RemoveXlaMarkTensorOpsPass
    : public mlir::OperationPass<mlir::func::FuncOp> {
 public:
  explicit RemoveXlaMarkTensorOpsPass()
      : mlir::OperationPass<mlir::func::FuncOp>::OperationPass(
            mlir::TypeID::get<RemoveXlaMarkTensorOpsPass>()) {}

  ~RemoveXlaMarkTensorOpsPass() override = default;

  void runOnOperation() override {
    mlir::func::FuncOp func_op = getOperation();
    llvm::SmallVector<mlir::Operation*> ops_to_erase;

    for (auto op : func_op.getOps<mlir::stablehlo::CustomCallOp>()) {
      if (!IsXlaMarkTensorOp(op.getOperation())) {
        continue;
      }
      mlir::Value original_value = op.getOperand(0);

      for (mlir::Value result : op.getResults()) {
        result.replaceAllUsesWith(original_value);
      }
    }

    // The unused custom_call ops will be eliminated with canonicalizer.
  }

  mlir::StringRef getName() const override {
    return llvm::getTypeName<RemoveXlaMarkTensorOpsPass>();
  }

  std::unique_ptr<mlir::Pass> clonePass() const override {
    return std::make_unique<RemoveXlaMarkTensorOpsPass>(*this);
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateBuildStableHLOCompositePass() {
  return std::make_unique<BuildStableHLOCompositePass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateRemoveXlaMarkTensorOpsPass() {
  return std::make_unique<RemoveXlaMarkTensorOpsPass>();
}

}  // namespace runtime
}  // namespace torch_xla
