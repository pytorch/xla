#include "init_python_bindings.h"

#include "module.h"
#include "passes/eval_static_size.h"
#include "passes/replace_in_place_ops.h"
#include "passes/replace_untraced_operators.h"
#include "passes/threshold_backward_peephole.h"
#include "tensorflow/compiler/xla/xla_client/metrics.h"
#include "torch/csrc/autograd/utils/wrap_outputs.h"
#include "torch_util.h"
#include "translator.h"

namespace torch_xla {
namespace {

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState* state = nullptr;
};

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
  m.def("_xla_sync_multi",
        [](const std::vector<std::shared_ptr<XLATensor>>& tensors) {
          NoGilSection nogil;
          XLATensor::ApplyPendingGraph(tensors, /*apply_context=*/nullptr);
        });
  m.def("_xla_to_tensors",
        [](const std::vector<std::shared_ptr<XLATensor>>& tensors) {
          std::vector<at::Tensor> result;
          {
            NoGilSection nogil;
            result = XLATensor::GetTensors(tensors);
          }
          return result;
        });
  m.def("_xla_create_tensors",
        [](const std::vector<torch::autograd::Variable>& tensors,
           const std::vector<std::string>& devices) {
          std::vector<std::shared_ptr<XLATensor>> result;
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
  py::class_<XLATensor, std::shared_ptr<XLATensor>>(m, "XLATensor")
      .def(py::init(
               [](torch::autograd::Variable tensor, const std::string& device) {
                 return XLATensor::Create(tensor,
                                          XLATensor::DeviceFromString(device));
               }),
           py::arg("tensor"), py::arg("device") = "")
      .def("to_tensor", [](XLATensor& s) { return s.ToTensor(); })
      .def("size", [](const XLATensor& s) { return s.Size(); })
      .def("device",
           [](const XLATensor& s) { return s.GetDevice().ToString(); })
      .def("__add__", [](std::shared_ptr<XLATensor> self,
                         XLATensor& other) { return self->add(other, 1.0); })
      .def("add", [](std::shared_ptr<XLATensor> self, double alpha,
                     XLATensor& other) { return self->add(other, alpha); })
      .def("add_",
           [](std::shared_ptr<XLATensor> self, double alpha, XLATensor& other) {
             self->add_(other, alpha);
             return self;
           })
      .def("add_",
           [](std::shared_ptr<XLATensor> self, XLATensor& other) {
             self->add_(other, 1.);
             return self;
           })
      .def(
          "__mul__",
          [](std::shared_ptr<XLATensor> self, XLATensor& other) {
            return self->mul(other);
          },
          py::arg("other"))
      .def("__mul__", [](std::shared_ptr<XLATensor> self,
                         double other) { return self->mul(other); })
      .def(
          "mul",
          [](std::shared_ptr<XLATensor> self, XLATensor& other) {
            return self->mul(other);
          },
          py::arg("other"))
      .def("mul", [](std::shared_ptr<XLATensor> self,
                     double other) { return self->mul(other); })
      .def(
          "mul_",
          [](std::shared_ptr<XLATensor> self, XLATensor& other) {
            self->mul_(other);
            return self;
          },
          py::arg("other"))
      .def("mul_",
           [](std::shared_ptr<XLATensor> self, double other) {
             self->mul_(other);
             return self;
           })
      .def(
          "__div__",
          [](std::shared_ptr<XLATensor> self, XLATensor& other) {
            return self->div(other);
          },
          py::arg("other"))
      .def("__div__", [](std::shared_ptr<XLATensor> self,
                         double other) { return self->div(other); })
      .def(
          "__truediv__",
          [](std::shared_ptr<XLATensor> self, XLATensor& other) {
            return self->div(other);
          },
          py::arg("other"))
      .def("__truediv__", [](std::shared_ptr<XLATensor> self,
                             double other) { return self->div(other); })
      .def("addcdiv_",
           [](std::shared_ptr<XLATensor> self, double alpha, XLATensor& tensor1,
              XLATensor& tensor2) {
             self->addcdiv_(alpha, tensor1, tensor2);
             return self;
           })
      .def("addcmul_",
           [](std::shared_ptr<XLATensor> self, double alpha, XLATensor& tensor1,
              XLATensor& tensor2) {
             self->addcmul_(alpha, tensor1, tensor2);
             return self;
           })
      .def("t", [](std::shared_ptr<XLATensor> self) { return self->t(); })
      .def("view",
           [](std::shared_ptr<XLATensor> self, py::args args) {
             std::vector<xla::int64> output_sizes;
             for (const auto& output_dim_size : args) {
               output_sizes.push_back(output_dim_size.cast<xla::int64>());
             }
             return self->view(output_sizes);
           })
      .def("cross_replica_sum",
           [](std::shared_ptr<XLATensor> self, const py::list& groups) {
             std::vector<std::vector<xla::int64>> crs_groups;
             for (auto& group : groups) {
               crs_groups.emplace_back();
               for (auto& replica_id : group.cast<py::list>()) {
                 crs_groups.back().push_back(replica_id.cast<xla::int64>());
               }
             }
             return self->cross_replica_sum(crs_groups);
           })
      .def("zero_",
           [](std::shared_ptr<XLATensor> self) {
             self->zero_();
             return self;
           })
      .def("detach_",
           [](std::shared_ptr<XLATensor> self) {
             self->detach_();
             return self;
           })
      .def_property_readonly(
          "data",
          [](std::shared_ptr<XLATensor> self) {
            return py::cast<std::shared_ptr<XLATensor>>(self->Clone());
          })
      .def_property_readonly(
          "dtype",
          [](std::shared_ptr<XLATensor> self) {
            return py::cast<py::object>(
                torch::autograd::utils::wrap(torch::getDtype(self->dtype())));
          })
      .def_property_readonly("is_leaf", [](const XLATensor&) { return true; })
      .def_property_readonly(
          "grad",
          [](XLATensor& m) -> py::object {
            if (m.grad() == nullptr) {
              return py::none();
            } else {
              return py::cast<std::shared_ptr<XLATensor>>(m.grad());
            }
          })
      .def("__repr__", [](XLATensor& m) {
        std::ostringstream s;
        s << m.ToTensor();
        return s.str();
      });
  m.def("relu", [](std::shared_ptr<XLATensor> self) { return self->relu(); });
  m.def(
      "conv2d",
      [](std::shared_ptr<XLATensor> self, std::shared_ptr<XLATensor> weight,
         std::shared_ptr<XLATensor> bias, int stride, int padding,
         bool use_full_conv_precision) {
        return self->conv2d(weight, bias, stride, padding,
                            use_full_conv_precision);
      },
      py::arg("input"), py::arg("weight"), py::arg("bias") = nullptr,
      py::arg("stride") = 1, py::arg("padding") = 0,
      py::arg("use_full_conv_precision") = false);
  m.def(
      "max_pool2d",
      [](std::shared_ptr<XLATensor> self, int kernel_size, int stride,
         int padding) {
        return self->max_pool2d(kernel_size, stride, padding);
      },
      py::arg("input"), py::arg("kernel_size"), py::arg("stride") = 1,
      py::arg("padding") = 0);
}

}  // namespace

void InitXlaBindings(py::module m) {
  InitXlaModuleBindings(m);
  InitXlaPassesBindings(m);
  InitXlaTensorBindings(m);
}

}  // namespace torch_xla

PYBIND11_MODULE(_XLAC, m) { torch_xla::InitXlaBindings(m); }
