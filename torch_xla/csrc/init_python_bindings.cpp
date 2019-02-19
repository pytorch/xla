#include "torch_xla/csrc/init_python_bindings.h"

#include <string>
#include <vector>

#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch/csrc/autograd/variable.h"
#include "torch_xla/csrc/aten_xla_bridge.h"
#include "torch_xla/csrc/aten_xla_type.h"
#include "torch_xla/csrc/ir_dump_util.h"
#include "torch_xla/csrc/module.h"
#include "torch_xla/csrc/passes/eval_static_size.h"
#include "torch_xla/csrc/passes/replace_in_place_ops.h"
#include "torch_xla/csrc/passes/replace_untraced_operators.h"
#include "torch_xla/csrc/passes/threshold_backward_peephole.h"
#include "torch_xla/csrc/torch_util.h"
#include "torch_xla/csrc/translator.h"

namespace torch_xla {
namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

std::string GetTensorsDump(
    const std::vector<at::Tensor>& tensors,
    const std::function<std::string(
        tensorflow::gtl::ArraySlice<const ir::Node* const>)>& coverter) {
  std::vector<const ir::Node*> nodes;
  std::vector<ir::Value> values;
  for (auto& tensor : tensors) {
    XLATensor xtensor = bridge::GetXlaTensor(ToTensor(tensor));
    values.push_back(xtensor.GetIrValue());
    nodes.push_back(values.back().node.get());
  }
  return coverter(nodes);
}

void InitXlaModuleBindings(py::module m) {
  py::class_<XlaModule, std::shared_ptr<XlaModule>>(m, "XlaModule")
      .def(py::init([](const std::shared_ptr<torch::jit::script::Module> module,
                       bool use_full_conv_precision, bool differentiate) {
             return std::make_shared<XlaModule>(module, use_full_conv_precision,
                                                differentiate);
           }),
           py::arg("module"), py::arg("use_full_conv_precision") = false,
           py::arg("differentiate") = true)
      .def("__call__",
           [](XlaModule& xla_module, py::args args) -> py::object {
             auto inputs = XlaCreateTensorList(args);
             XlaModule::TensorBatchVector outputs;
             {
               NoGilSection nogil;
               outputs = xla_module.forward(inputs);
             }
             return XlaPackTensorList(outputs);
           })
      .def("backward",
           [](XlaModule& xla_module, py::args args) {
             auto inputs = XlaCreateTensorList(args);
             NoGilSection nogil;
             xla_module.backward(inputs);
           })
      .def("set_input_gradients",
           [](XlaModule& xla_module, const py::list& gradient_list) {
             std::vector<at::Tensor> gradients;
             for (auto& gradient : gradient_list) {
               gradients.push_back(gradient.cast<at::Tensor>());
             }
             xla_module.SetInputGradientsForFusion(std::move(gradients));
           })
      .def("parameters",
           [](XlaModule& xla_module) { return xla_module.parameters(); })
      .def("parameters_buffers", [](XlaModule& xla_module) {
        return xla_module.parameters_buffers();
      });
  m.def("_register_aten_types", []() { AtenXlaType::RegisterAtenTypes(); });
  m.def("_get_xla_tensor", [](const at::Tensor& tensor) -> XLATensor {
    return bridge::GetXlaTensor(ToTensor(tensor));
  });
  m.def("_get_xla_tensors_dot",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter =
              [](tensorflow::gtl::ArraySlice<const ir::Node* const> nodes) {
                return ir::DumpUtil::ToDot(nodes);
              };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_get_xla_tensors_text",
        [](const std::vector<at::Tensor>& tensors) -> std::string {
          auto coverter =
              [](tensorflow::gtl::ArraySlice<const ir::Node* const> nodes) {
                return ir::DumpUtil::ToText(nodes);
              };
          return GetTensorsDump(tensors, coverter);
        });
  m.def("_xla_sync_multi", [](std::vector<XLATensor>& tensors) {
    NoGilSection nogil;
    XLATensor::ApplyPendingGraph(&tensors, /*apply_context=*/nullptr);
  });
  m.def("_xla_to_tensors", [](std::vector<XLATensor>& tensors) {
    std::vector<at::Tensor> result;
    {
      NoGilSection nogil;
      std::vector<at::Tensor> raw_tensors =
          XLATensor::GetTensors(&tensors, /*writeable=*/nullptr);
      for (size_t i = 0; i < tensors.size(); ++i) {
        result.push_back(torch::autograd::make_variable(
            std::move(raw_tensors[i]), tensors[i].RequiresGrad()));
      }
    }
    return result;
  });
  m.def("_xla_create_tensors", [](const std::vector<at::Tensor>& tensors,
                                  const std::vector<std::string>& devices) {
    std::vector<XLATensor> result;
    {
      NoGilSection nogil;
      result = XLATensor::CreateTensors(tensors, devices);
    }
    return result;
  });
  m.def("_xla_counter_value", [](const std::string& name) -> py::object {
    xla::metrics::CounterData* data = xla::metrics::GetCounter(name);
    return data != nullptr ? py::cast<int64_t>(data->Value()) : py::none();
  });
  m.def("_xla_metrics_report",
        []() { return xla::metrics::CreateMetricReport(); });
}

void InitXlaPassesBindings(py::module m) {
  m.def("_jit_pass_eval_static_size", EvalStaticSize);
  m.def("_jit_pass_replace_untraced_operators", ReplaceUntracedOperators);
  m.def("_jit_pass_threshold_backward_peephole", ThresholdBackwardPeephole);
  m.def("_jit_pass_replace_in_place_ops", ReplaceInPlaceOps);
}

void InitXlaTensorBindings(py::module m) {
  py::class_<XLATensor>(m, "XLATensor")
      .def(py::init([](const torch::autograd::Variable& tensor,
                       const std::string& device) {
             return XLATensor::Create(tensor.data().clone(), Device(device),
                                      tensor.requires_grad());
           }),
           py::arg("tensor"), py::arg("device") = "")
      .def("to_tensor",
           [](XLATensor& s) {
             return torch::autograd::make_variable(s.ToTensor(),
                                                   s.RequiresGrad());
           })
      .def("size", [](const XLATensor& s) { return s.DimensionSizes(); })
      .def("device",
           [](const XLATensor& s) { return s.GetDevice().ToString(); })
      .def("__add__",
           [](const XLATensor& self, XLATensor& other) {
             return XLATensor::add(self, other, 1.0);
           })
      .def("add",
           [](const XLATensor& self, double alpha, const XLATensor& other) {
             return XLATensor::add(self, other, alpha);
           })
      .def("add_",
           [](XLATensor self, double alpha, const XLATensor& other) {
             XLATensor::add_(self, other, alpha);
             return self;
           })
      .def("add_",
           [](XLATensor self, const XLATensor& other) {
             XLATensor::add_(self, other, 1.0);
             return self;
           })
      .def(
          "__mul__",
          [](const XLATensor& self, const XLATensor& other) {
            return XLATensor::mul(self, other);
          },
          py::arg("other"))
      .def("__mul__", [](const XLATensor& self,
                         double other) { return XLATensor::mul(self, other); })
      .def(
          "mul",
          [](const XLATensor& self, const XLATensor& other) {
            return XLATensor::mul(self, other);
          },
          py::arg("other"))
      .def("mul", [](const XLATensor& self,
                     double other) { return XLATensor::mul(self, other); })
      .def(
          "mul_",
          [](XLATensor self, const XLATensor& other) {
            XLATensor::mul_(self, other);
            return self;
          },
          py::arg("other"))
      .def("mul_",
           [](XLATensor self, double other) {
             XLATensor::mul_(self, other);
             return self;
           })
      .def(
          "__div__",
          [](const XLATensor& self, const XLATensor& other) {
            return XLATensor::div(self, other);
          },
          py::arg("other"))
      .def("__div__", [](const XLATensor& self,
                         double other) { return XLATensor::div(self, other); })
      .def(
          "__truediv__",
          [](const XLATensor& self, const XLATensor& other) {
            return XLATensor::div(self, other);
          },
          py::arg("other"))
      .def("__truediv__",
           [](const XLATensor& self, double other) {
             return XLATensor::div(self, other);
           })
      .def("t", [](const XLATensor& self) { return XLATensor::t(self); })
      .def("view",
           [](const XLATensor& self, py::args args) {
             std::vector<xla::int64> output_sizes;
             for (const auto& output_dim_size : args) {
               output_sizes.push_back(output_dim_size.cast<xla::int64>());
             }
             return XLATensor::view(self, output_sizes);
           })
      .def("log_softmax",
           [](const XLATensor& self, int dim) {
             return XLATensor::log_softmax(self, dim);
           })
      .def("cross_replica_sum",
           [](const XLATensor& self, const py::list& groups) {
             std::vector<std::vector<xla::int64>> crs_groups;
             for (auto& group : groups) {
               crs_groups.emplace_back();
               for (auto& replica_id : group.cast<py::list>()) {
                 crs_groups.back().push_back(replica_id.cast<xla::int64>());
               }
             }
             return self.cross_replica_sum(crs_groups);
           })
      .def("zero_",
           [](XLATensor self) {
             XLATensor::zero_(self);
             return self;
           })
      .def("detach_",
           [](XLATensor self) {
             self.detach_();
             return self;
           })
      .def("size",
           [](const XLATensor& self, int dim) { return self.size(dim); })
      .def_property_readonly(
          "data",
          [](const XLATensor& self) { return py::cast<XLATensor>(self); })
      .def_property_readonly(
          "dtype",
          [](const XLATensor& self) {
            return py::cast<py::object>(
                torch::autograd::utils::wrap(torch::getDtype(self.dtype())));
          })
      .def_property_readonly("is_leaf", [](const XLATensor&) { return true; })
      .def_property_readonly("grad",
                             [](XLATensor& m) -> py::object {
                               auto grad = m.grad();
                               if (!grad) {
                                 return py::none();
                               } else {
                                 return py::cast<XLATensor>(*grad);
                               }
                             })
      .def("__repr__", [](XLATensor& m) {
        std::ostringstream s;
        s << m.ToTensor();
        return s.str();
      });
  m.def("relu", [](const XLATensor& self) { return XLATensor::relu(self); });
  m.def("threshold", [](const XLATensor& self, float threshold, float value) {
    return XLATensor::threshold(self, threshold, value);
  });
  m.def(
      "conv2d",
      [](const XLATensor& self, const XLATensor& weight, int stride,
         int padding, bool use_full_conv_precision) {
        std::vector<xla::int64> stride_2d(2, stride);
        std::vector<xla::int64> padding_2d(2, padding);
        return XLATensor::conv2d(self, weight, stride_2d, padding_2d,
                                 use_full_conv_precision);
      },
      py::arg("input"), py::arg("weight"), py::arg("stride") = 1,
      py::arg("padding") = 0, py::arg("use_full_conv_precision") = false);
  m.def(
      "conv2d",
      [](const XLATensor& self, const XLATensor& weight, const XLATensor& bias,
         int stride, int padding, bool use_full_conv_precision) {
        std::vector<xla::int64> stride_2d(2, stride);
        std::vector<xla::int64> padding_2d(2, padding);
        return XLATensor::conv2d(self, weight, bias, stride_2d, padding_2d,
                                 use_full_conv_precision);
      },
      py::arg("input"), py::arg("weight"), py::arg("bias"),
      py::arg("stride") = 1, py::arg("padding") = 0,
      py::arg("use_full_conv_precision") = false);
  m.def(
      "conv2d",
      [](const XLATensor& self, const XLATensor& weight,
         const std::vector<xla::int64>& stride,
         const std::vector<xla::int64>& padding, bool use_full_conv_precision) {
        return XLATensor::conv2d(self, weight, stride, padding,
                                 use_full_conv_precision);
      },
      py::arg("input"), py::arg("weight"),
      py::arg("stride") = std::vector<xla::int64>{1, 1},
      py::arg("padding") = std::vector<xla::int64>{0, 0},
      py::arg("use_full_conv_precision") = false);

  m.def(
      "conv2d",
      [](const XLATensor& self, const XLATensor& weight, const XLATensor& bias,
         const std::vector<xla::int64>& stride,
         const std::vector<xla::int64>& padding, bool use_full_conv_precision) {
        return XLATensor::conv2d(self, weight, bias, stride, padding,
                                 use_full_conv_precision);
      },
      py::arg("input"), py::arg("weight"), py::arg("bias"),
      py::arg("stride") = std::vector<xla::int64>{1, 1},
      py::arg("padding") = std::vector<xla::int64>{0, 0},
      py::arg("use_full_conv_precision") = false);
  m.def(
      "addmm",
      [](XLATensor& bias, const XLATensor& input, const XLATensor& weight,
         bool use_full_conv_precision) {
        return XLATensor::addmm(input, weight, bias, use_full_conv_precision);
      },
      py::arg("bias"), py::arg("input"), py::arg("weight"),
      py::arg("use_full_conv_precision") = false);
  m.def(
      "max_pool2d",
      [](const XLATensor& self, int kernel_size, int stride, int padding) {
        return XLATensor::max_pool2d(self, {kernel_size, kernel_size},
                                     {stride, stride}, {padding, padding});
      },
      py::arg("input"), py::arg("kernel_size"), py::arg("stride") = 1,
      py::arg("padding") = 0);
  m.def(
      "max_pool2d",
      [](const XLATensor& self, const std::vector<xla::int64>& kernel_size,
         const std::vector<xla::int64>& stride,
         const std::vector<xla::int64>& padding) {
        return XLATensor::max_pool2d(self, kernel_size, stride, padding);
      },
      py::arg("input"), py::arg("kernel_size"),
      py::arg("stride") = std::vector<xla::int64>{1, 1},
      py::arg("padding") = std::vector<xla::int64>{0, 0});
  m.def(
      "avg_pool2d",
      [](const XLATensor& self, int kernel_size, int stride, int padding,
         bool count_include_pad) {
        return XLATensor::avg_pool2d(self, {kernel_size, kernel_size},
                                     {stride, stride}, {padding, padding},
                                     count_include_pad);
      },
      py::arg("input"), py::arg("kernel_size"), py::arg("stride") = 1,
      py::arg("padding") = 0, py::arg("count_include_pad") = true);
  m.def(
      "avg_pool2d",
      [](const XLATensor& self, const std::vector<xla::int64>& kernel_size,
         const std::vector<xla::int64>& stride,
         const std::vector<xla::int64>& padding, bool count_include_pad) {
        return XLATensor::avg_pool2d(self, kernel_size, stride, padding,
                                     count_include_pad);
      },
      py::arg("input"), py::arg("kernel_size"),
      py::arg("stride") = std::vector<xla::int64>{1, 1},
      py::arg("padding") = std::vector<xla::int64>{0, 0},
      py::arg("count_include_pad") = true);
}

}  // namespace

void InitXlaBindings(py::module m) {
  InitXlaModuleBindings(m);
  InitXlaPassesBindings(m);
  InitXlaTensorBindings(m);
}

}  // namespace torch_xla

PYBIND11_MODULE(_XLAC, m) { torch_xla::InitXlaBindings(m); }
