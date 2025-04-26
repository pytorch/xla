#include "torch_xla/csrc/runtime/stablehlo_composite_helper.h"

#include <limits>
#include <string>
#include <tuple>
#include <unordered_map>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/LogicalResult.h"
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
      mlir::dyn_cast<mlir::StringAttr>(op->getAttr("call_target_name"));
  if (target_name == nullptr || target_name.str() != "xla_mark_tensor") {
    return false;
  }
  return true;
}

struct BoundaryMetadata {
  std::string name;
  std::string id;
  int64_t pos;
  bool is_input;
  std::unordered_map<std::string, json> attrs;

  auto boundary_key() const { return absl::StrCat(name, "__@@__", id); }

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
        CopyJsonValue(j, "id", json::value_t::string, metadata.id) &&
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
      llvm::DenseMap<const mlir::Operation*, size_t> op_order_map =
          BuildOpOrderMap(func_op);
      std::unordered_map<std::string, llvm::SmallVector<mlir::Operation*>>
          boundary_output_ops_map = BuildBoundaryOutputOpsMap(func_op);

      for (const auto& [unused, ops] : boundary_output_ops_map) {
        if (mlir::failed(BuildStableHLOComposite(ops, op_order_map))) {
          func_op.emitError() << "failed to build composite.";
          return signalPassFailure();
        }
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
  llvm::DenseMap<const mlir::Operation*, size_t> BuildOpOrderMap(
      mlir::func::FuncOp func_op) const {
    llvm::DenseMap<const mlir::Operation*, size_t> op_order_map;
    for (const auto& op : llvm::enumerate(func_op.getOps())) {
      op_order_map[&op.value()] = op.index();
    }
    return op_order_map;
  }

  std::unordered_map<std::string, llvm::SmallVector<mlir::Operation*>>
  BuildBoundaryOutputOpsMap(mlir::func::FuncOp func_op) {
    std::unordered_map<std::string, llvm::SmallVector<mlir::Operation*>>
        boundary_output_ops;

    for (auto op : func_op.getOps<mlir::stablehlo::CustomCallOp>()) {
      auto metadata_or = GetBoundaryMetadata(op);
      if (mlir::failed(metadata_or)) {
        continue;
      }

      std::unique_ptr<BoundaryMetadata> metadata = std::move(*metadata_or);
      if (metadata == nullptr || metadata->is_input) {
        continue;
      }

      auto& output_ops = boundary_output_ops[metadata->boundary_key()];
      if (metadata->pos >= output_ops.size()) {
        output_ops.resize(metadata->pos + 1, nullptr);
      }
      output_ops[metadata->pos] = op.getOperation();
    }
    return boundary_output_ops;
  }

  mlir::FailureOr<std::unique_ptr<BoundaryMetadata>> GetBoundaryMetadata(
      mlir::Operation* op) {
    if (!IsXlaMarkTensorOp(op)) {
      return mlir::FailureOr(nullptr);
    }
    auto backend_config =
        mlir::dyn_cast<mlir::StringAttr>(op->getAttr("backend_config"));
    if (backend_config == nullptr) {
      return mlir::FailureOr(nullptr);
    }
    std::unique_ptr<BoundaryMetadata> metadata =
        BoundaryMetadata::Parse(backend_config);
    if (metadata == nullptr) {
      return op->emitError() << "invalid boundary metadata JSON.";
    }
    return metadata;
  }

  mlir::FailureOr<mlir::Attribute> BuildAttrFromJson(mlir::OpBuilder& builder,
                                                     mlir::Operation* op,
                                                     const json& json_value) {
    switch (json_value.type()) {
      case json::value_t::number_integer:
      case json::value_t::number_unsigned:
        return builder.getI64IntegerAttr(json_value.template get<int64_t>());
      case json::value_t::number_float:
        return builder.getF32FloatAttr(json_value.template get<float>());
      case json::value_t::boolean:
        return builder.getBoolAttr(json_value.template get<bool>());
      case json::value_t::string:
        return builder.getStringAttr(json_value.template get<std::string>());
      case json::value_t::array: {
        if (json_value.empty()) {
          return builder.getArrayAttr({});
        }
        auto get_json_type = [](const json& j) {
          auto ty = j.type();
          if (ty == json::value_t::number_unsigned) {
            return json::value_t::number_integer;
          }
          return ty;
        };

        auto head_type = get_json_type(json_value[0]);
        bool is_homogeneous = llvm::all_of(json_value, [&](auto& el) {
          return get_json_type(el) == head_type;
        });
        if (!is_homogeneous) {
          return op->emitError()
                 << "invalid JSON to MLIR, arrays must be homogeneous";
        }

        switch (head_type) {
          case json::value_t::number_integer:
            return builder.getI64TensorAttr(
                json_value.template get<llvm::SmallVector<int64_t>>());
          case json::value_t::number_float:
            return mlir::DenseFPElementsAttr::get(
                mlir::RankedTensorType::get(json_value.size(),
                                            builder.getF32Type()),
                json_value.template get<llvm::SmallVector<float>>());
          case json::value_t::boolean:
            return mlir::DenseIntElementsAttr::get(
                mlir::RankedTensorType::get(json_value.size(),
                                            builder.getI1Type()),
                json_value.template get<llvm::SmallVector<bool>>());
          default:
            return op->emitError()
                   << "invalid JSON to MLIR: invalid array type. arrays must "
                      "be "
                      "1-D homogeneous arrays of supported primitive types";
        }
      }
      default:
        return op->emitError()
               << "invalid JSON to MLIR: unsupported json value type";
    }
  }

  mlir::FailureOr<mlir::DictionaryAttr> BuildDictionaryAttrFromJsonMap(
      mlir::OpBuilder& builder, mlir::Operation* op,
      const std::unordered_map<std::string, json>& json_map) {
    llvm::SmallVector<mlir::NamedAttribute> named_attrs;
    for (auto& [key, j] : json_map) {
      mlir::FailureOr<mlir::Attribute> attribute_or =
          BuildAttrFromJson(builder, op, j);
      if (mlir::failed(attribute_or)) {
        return mlir::failure();
      }
      named_attrs.push_back({builder.getStringAttr(key), *attribute_or});
    }
    return builder.getDictionaryAttr(named_attrs);
  }

  mlir::LogicalResult BuildStableHLOComposite(
      const llvm::SmallVector<mlir::Operation*>& output_ops,
      const llvm::DenseMap<const mlir::Operation*, size_t>& op_order_map) {
    if (output_ops.empty()) {
      return mlir::success();
    }

    // Get the output op with minimum order num as the representative.
    mlir::Operation* first_output_op = output_ops[0];
    for (mlir::Operation* op : output_ops) {
      if (op_order_map.at(op) < op_order_map.at(first_output_op)) {
        first_output_op = op;
      }
    }

    auto metadata_or = GetBoundaryMetadata(first_output_op);
    if (mlir::failed(metadata_or)) {
      return mlir::failure();
    }

    std::unique_ptr<BoundaryMetadata> metadata = std::move(*metadata_or);
    if (metadata == nullptr || metadata->is_input) {
      // There should always be a valid boundary output metadata associated with
      // each op in output_ops.
      return mlir::failure();
    }

    auto args_ops_or =
        GetBoundaryArgsAndOps(output_ops, *metadata, op_order_map);
    if (mlir::failed(args_ops_or)) {
      return mlir::failure();
    }

    auto [args, impl_ops] = *args_ops_or;

    mlir::func::FuncOp impl_func = BuildStableHLOCompositeImplFunc(
        output_ops, absl::StrCat(metadata->name, ".impl"), args, impl_ops);
    mlir::FailureOr<mlir::Operation*> composite_op_or =
        BuildStableHLOCompositeOp(first_output_op, impl_func, args, *metadata);
    if (mlir::failed(composite_op_or)) {
      return mlir::failure();
    }
    mlir::Operation* composite_op = *composite_op_or;

    // Updates all users of this op's result(s) to use the results(s) of impl
    // func call.
    size_t composite_result_i = 0;
    for (mlir::Operation* op : output_ops) {
      for (size_t i = 0; i < op->getNumResults(); ++i) {
        mlir::OpResult result = op->getResult(i);
        result.replaceAllUsesWith(
            composite_op->getResult(composite_result_i++));
      }
    }

    if (!mlir::sortTopologically(composite_op->getBlock())) {
      composite_op->emitError()
          << "The graph is not acyclic after BuildStableHLOCompositePass pass.";
      return mlir::failure();
    }
    // The unused impl_ops will be eliminated with canonicalizer.
    return mlir::success();
  }

  mlir::FailureOr<std::pair<llvm::SmallVector<mlir::Value>,
                            llvm::SmallVector<mlir::Operation*>>>
  GetBoundaryArgsAndOps(
      const llvm::SmallVector<mlir::Operation*> boundary_output_ops,
      const BoundaryMetadata& metadata,
      const llvm::DenseMap<const mlir::Operation*, size_t>& op_order_map) {
    llvm::SetVector<mlir::Operation*> impl_ops_setvec;
    llvm::SetVector<std::pair<mlir::Value, int64_t>> arg_pos_setvec;
    llvm::SmallVector<mlir::Operation*> processing(boundary_output_ops.begin(),
                                                   boundary_output_ops.end());

    // Reverse graph traversal: from boundary output op to boundary input op,
    // global function arg, or stablehlo constant.
    while (!processing.empty()) {
      mlir::Operation* curr_op = processing.back();
      processing.pop_back();
      if (impl_ops_setvec.contains(curr_op)) {
        continue;
      }

      auto curr_metadata_or = GetBoundaryMetadata(curr_op);
      if (mlir::failed(curr_metadata_or)) {
        return mlir::failure();
      }
      std::unique_ptr<BoundaryMetadata> curr_metadata =
          std::move(*curr_metadata_or);
      if (curr_metadata != nullptr) {
        if (curr_metadata->is_input &&
            curr_metadata->boundary_key() == metadata.boundary_key()) {
          // Terminal condition: boundary input op.
          arg_pos_setvec.insert(
              {mlir::dyn_cast<mlir::Value>(curr_op->getResult(0)),
               curr_metadata->pos});
          continue;
        }
      }

      impl_ops_setvec.insert(curr_op);
      for (mlir::Value value : curr_op->getOperands()) {
        mlir::Operation* def_op = value.getDefiningOp();
        if (def_op == nullptr) {
          // Terminal condition: global function arg
          arg_pos_setvec.insert({value, std::numeric_limits<int64_t>::max()});
        } else if (llvm::isa<mlir::stablehlo::ConstantOp>(def_op)) {
          // Terminal condition: constant
          impl_ops_setvec.insert(def_op);
        } else {
          processing.push_back(def_op);
        }
      }
    }
    // Sorts all ops within the boundary by their line numbers in the input
    // MLIR. The ops will be duplicated to the impl function following this
    // order.
    llvm::SmallVector<mlir::Operation*> impl_ops = impl_ops_setvec.takeVector();
    for (auto& op : impl_ops) {
      if (!op_order_map.contains(op)) {
        return op->emitError()
               << "does not have a ordering number in its outer func.";
      }
    }
    std::sort(impl_ops.begin(), impl_ops.end(),
              [&op_order_map](const auto& a, const auto& b) {
                return op_order_map.at(a) < op_order_map.at(b);
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

    return std::make_pair(std::move(args), std::move(impl_ops));
  }

  mlir::func::FuncOp BuildStableHLOCompositeImplFunc(
      const llvm::SmallVector<mlir::Operation*> boundary_output_ops,
      llvm::StringRef func_name, const llvm::SmallVector<mlir::Value>& args,
      const llvm::SmallVector<mlir::Operation*>& impl_ops) {
    mlir::ModuleOp module_op = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::OpBuilder builder(context);

    // Creates composite impl function and duplicates all ops within the
    // boundary in the function.
    llvm::SmallVector<mlir::Location> arg_locs;
    llvm::SmallVector<mlir::Type> arg_types;
    for (auto& arg : args) {
      arg_types.push_back(arg.getType());
      arg_locs.push_back(arg.getLoc());
    }
    llvm::SmallVector<mlir::Type> result_types;
    for (mlir::Operation* op : boundary_output_ops) {
      result_types.append(op->getResultTypes().begin(),
                          op->getResultTypes().end());
    }

    mlir::func::FuncOp impl_func = builder.create<mlir::func::FuncOp>(
        module_op.getLoc(), func_name,
        mlir::FunctionType::get(context, arg_types, result_types));
    mlir::IRMapping mapping;
    builder.createBlock(&impl_func.getBody(), impl_func.begin(), arg_types,
                        arg_locs);
    for (const auto& arg : llvm::enumerate(args)) {
      mapping.map(arg.value(), impl_func.getArgument(arg.index()));
    }
    for (mlir::Operation* original_op : impl_ops) {
      mlir::Operation* cloned_op = builder.clone(*original_op, mapping);
      mapping.map(original_op, cloned_op);
    }

    llvm::SmallVector<mlir::Value> results;
    for (mlir::Operation* op : boundary_output_ops) {
      results.append(mapping.lookup(op)->getResults().begin(),
                     mapping.lookup(op)->getResults().end());
    }
    builder.create<mlir::func::ReturnOp>(impl_func.getBody().getLoc(), results);

    // Adds the new function to symbol table.
    mlir::SymbolTable symbol_table(module_op);
    impl_func.setPrivate();
    symbol_table.insert(impl_func);

    return impl_func;
  }

  mlir::FailureOr<mlir::Operation*> BuildStableHLOCompositeOp(
      mlir::Operation* boundary_output_op, mlir::func::FuncOp impl_func,
      const llvm::SmallVector<mlir::Value>& args,
      const BoundaryMetadata& metadata) {
    mlir::ModuleOp module_op = getOperation();
    mlir::MLIRContext* context = &getContext();
    mlir::OpBuilder builder(context);

    mlir::FailureOr<mlir::DictionaryAttr> attributes_or =
        BuildDictionaryAttrFromJsonMap(builder, boundary_output_op,
                                       metadata.attrs);
    if (mlir::failed(attributes_or)) {
      return boundary_output_op->emitError()
             << "failed to transform boundary attr "
                "JSON into composite attributes.";
    }

    // Creates and inserts composite call op.
    builder.setInsertionPointAfter(boundary_output_op);
    mlir::Operation* composite_op =
        builder.create<mlir::stablehlo::CompositeOp>(
            boundary_output_op->getLoc(),
            impl_func.getFunctionType().getResults(), args, metadata.name,
            *attributes_or, impl_func.getSymName());
    return composite_op;
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
