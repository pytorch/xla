#include "torch_util.h"

namespace torch_xla {

XlaModule::TensorBatchVector XlaCreateTensorList(const py::tuple& tuple) {
  XlaModule::TensorBatchVector result;
  result.reserve(tuple.size());
  for (auto& replica_tuple : tuple) {
    XlaModule::TensorBatchVector::value_type replica_result;
    for (auto& e : replica_tuple) {
      auto variable = py::cast<XLATensor>(e);
      replica_result.push_back(variable);
    }
    result.push_back(std::move(replica_result));
  }
  return result;
}

py::object XlaPackTensorList(const XlaModule::TensorBatchVector& outputs) {
  py::tuple tuple(outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    const auto& replica_outputs = outputs[i];
    py::tuple replica_tuple(replica_outputs.size());
    for (size_t j = 0; j < replica_outputs.size(); j++) {
      replica_tuple[j] = py::cast(replica_outputs[j]);
    }
    tuple[i] = replica_tuple;
  }
  return std::move(tuple);
}

}  // namespace torch_xla
