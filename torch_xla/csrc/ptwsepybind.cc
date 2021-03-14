#include <iostream>

#include "torch_xla/csrc/lazy_sentinel.h"

#include "tensorflow/compiler/xla/proxy_client/proxy_computation_client.h"
#include "tensorflow/compiler/xla/proxy_client/xla_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"

#include "torch/csrc/jit/python/pybind.h"

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace py = pybind11;

using namespace torch_xla;

namespace torch_xla {
void ptxla_StepMarker(const std::string &device_str,
                      const std::vector<std::string> &devices, bool wait);
std::vector<XLATensor>
ptxla_GetXlaTensors(const std::vector<at::Tensor> &tensors, bool want_all);
} // namespace torch_xla

namespace {
/**
 *
 *     /\
 *    /  \    _ __   ___  _ __  _   _ _ __ ___   ___  _   _  ___
 *   / /\ \  | '_ \ / _ \| '_ \| | | | '_ ` _ \ / _ \| | | |/ __|
 *  / ____ \ | | | | (_) | | | | |_| | | | | | | (_) | |_| |\__ \
 * /_/    \_\|_| |_|\___/|_| |_|\__, |_| |_| |_|\___/ \__,_||___/
 *                               __/ |
 *                              |___/
 */

bool unregister_on_shutdown = false;

std::shared_ptr<torch_xla::Sentinel> saved_sentinel{nullptr};
std::shared_ptr<xla::ComputationClientFactory> saved_computation_client_factory{
    nullptr};
std::shared_ptr<torch_xla::LazySentinel> our_sentinel{nullptr};

struct NoGilSection {
  NoGilSection() : state(PyEval_SaveThread()) {}
  ~NoGilSection() { PyEval_RestoreThread(state); }
  PyThreadState *state = nullptr;
};

/**
 * @brief Install the plugin
 */
void Initialize() {
  saved_sentinel = torch_xla::Sentinel::SetSentinel(
      std::make_shared<torch_xla::LazySentinel>());
  saved_computation_client_factory = xla::ComputationClient::SetFactory(
      std::make_shared<
          xla::TComputationClientFactory<xla::ProxyComputationClient>>());
  std::cout << "PTWSE Plugin Initialized." << std::endl;
}

/**
 * @brief Uninstall the plugin
 */
void Shutdown() {
  if (unregister_on_shutdown) {
    if (saved_computation_client_factory) {
      xla::ComputationClient::SetFactory(saved_computation_client_factory);
      saved_computation_client_factory.reset();
    }
    if (saved_sentinel) {
      torch_xla::Sentinel::SetSentinel(saved_sentinel);
      saved_sentinel.reset();
    }
  }
  std::cout << "PTWSE Plugin Shut down." << std::endl;
}

/**
 * Optional graph-pruning mechanism to reduce superfluous outputs
 */
void SetOutputs(const std::vector<at::Tensor> &output_tensors, bool append) {
  torch_xla::LazySentinel::SetOutputs(output_tensors, append);
}

/**
 * Plugin's outer step marker for "headless" mode
 */
bool PtwseStepMarker(const std::string &device_str,
                     const std::vector<std::string> &devices, bool wait) {
  torch_xla::ptxla_StepMarker(device_str, devices, wait);
  return torch_xla::LazySentinel::WasMarkStepOnProxy();
}

void SetDeviceProxy(const std::string &device,
                    const std::string &proxy_address) {
  // Currently just using the XlaComputationClient factory
  // TODO: Can move the proxy address into the factory
  std::shared_ptr<xla::ComputationClientFactory> client_factory =
      std::make_shared<xla::XlaComputationClientFactory>(device,
                                                         /*create_proxy=*/true);
  torch_xla::LazySentinel::SetDeviceProxy(device, proxy_address,
                                          std::move(client_factory));
}

} // anonymous namespace

PYBIND11_MODULE(_PTWSE, m) {
  m.doc() = ("pybind11 for PyTorch multi-device support.");
  m.def("_ptproxy_initialize", []() {
    NoGilSection gil;
    Initialize();
  });
  m.def("_ptproxy_shutdown", []() {
    NoGilSection gil;
    Shutdown();
  });
  m.def("_ptproxy_set_outputs",
        [](const std::vector<at::Tensor> &output_tensors, bool append) {
          SetOutputs(output_tensors, append);
        },
        py::arg("output_tensors"), py::arg("append") = false);
  m.def("_ptproxy_was_previous_mark_step_on_proxy",
        []() { return torch_xla::LazySentinel::WasMarkStepOnProxy(); });
  m.def("_ptproxy_is_initialized", []() {
    NoGilSection nogil;
    return torch_xla::LazySentinel::IsInitialized();
  });

  // IR Scope
  // Used for inserting scope naming into the nodes similar
  // in spirit to TensorFlow's variable_scope()
  m.def("_ptproxy_push_ir_scope", [](std::string scope) {
    torch_xla::ir::PythonPushScope(std::move(scope));
  });
  m.def("_ptproxy_pop_ir_scope", []() { torch_xla::ir::PythonPopScope(); });

  // Debug trap
  m.def("_ptproxy_trap", []() { raise(SIGTRAP); });

  // Step Marker
  m.def("_ptproxy_step_marker",
        [](const std::string &device, const std::vector<std::string> &devices,
           bool wait) -> bool {
          NoGilSection nogil;
          return PtwseStepMarker(device, devices, wait);
        },
        py::arg("device") = "", py::arg("devices"), py::arg("wait") = true);
}
