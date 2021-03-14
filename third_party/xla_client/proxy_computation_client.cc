#include "tensorflow/compiler/xla/xla_client/proxy_computation_client.h"
#include "tensorflow/compiler/xla/xla_client/color_output.h"
#include "tensorflow/compiler/xla/xla_client/global_data_handle_mapper.h"
#include "tensorflow/compiler/xla/xla_client/proxy_client_util.h"
#include "tensorflow/compiler/xla/xla_client/proxy_name.h"
#include "tensorflow/compiler/xla/xla_client/split_types.h"
#include "tensorflow/compiler/xla/xla_client/xla_computation_client.h"

#include <csignal>
#include <stdexcept>
#include <string>
#include <strstream>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/rpc/xla_service.grpc.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/xla_client/computation_client.h"
#include "tensorflow/compiler/xla/xla_client/thread_pool.h"
#include "tensorflow/compiler/xla/xla_client/xrt_computation_client.h"
#include "tensorflow/core/protobuf/tpu/topology.pb.h"

/**
 * TODO: Non-TF-linking portions of this to be moved to
 *       monolith-side after grpc boundary inserted
 */
using namespace tensorflow;
using namespace xla::torch_xla;

using Status = ::grpc::Status;

namespace xla {

namespace {

bool verbose = false;

/**
 * @brief Force always using the proxy server for everyting
 *        (i.e. delegate everything to the grpc_service_main app)
 */
bool always_use_proxy = false;
bool using_grpc_service_main_cpu = false;
bool throw_on_compile_fail = true;
bool verbose_pull = false;
bool verbose_transfer = false;
bool verbose_topology = false;
bool verbose_handle_mapping = false;
bool is_initialized = false;

}  // namespace

/**
 *  _____        _
 * |  __ \      | |
 * | |  | | __ _| |_  __ _
 * | |  | |/ _` | __|/ _` |
 * | |__| | (_| | |_| (_| |
 * |_____/ \__,_|\__|\__,_|
 *
 *
 */
ComputationClient::DataPtr ProxyComputationClient::CreateDataPlaceholder(
    std::string device, Shape shape) {
  if (ProxyName::is_proxy_device_name(device)) {
    std::string unproxy_device = ProxyName::unproxy_device_name(device);
    return std::make_shared<XrtData>(std::move(unproxy_device),
                                     std::move(shape));
  }
  return Super::CreateDataPlaceholder(std::move(device), std::move(shape));
}

ProxyComputationClient::ProxyComputationClient(
    Options options,
    std::unique_ptr<tensorflow::tpu::TopologyProto> topology_proto,
    XrtLocalService *service)
    : XrtComputationClient(std::move(options), std::move(topology_proto),
                           service),
      data_mapper_(std::make_shared<GlobalDataHandleMapper>()) {
  verbose = xla::sys_util::GetEnvBool("XLA_VERBOSE", false);
  ::setenv("XRT_MASTER_ALLOW_SAME_TASKS", "1", true);
  assert(!is_initialized);
  is_initialized = true;
}

void ProxyComputationClient::PrepareToExit() {
  client_manager_.PrepareToExit();
  XrtComputationClient::PrepareToExit();
  /// Do client manager again in case XrtComputationClient
  /// caused churn
  client_manager_.PrepareToExit();
}

bool ProxyComputationClient::IsInitialized() { return is_initialized; }

bool ProxyComputationClient::IsEnabled() {
  static bool enabled = xla::sys_util::GetEnvBool("XLA_PROXY_ENABLED", true);
  return enabled;
}

bool ProxyComputationClient::ShouldCloneDataForDevice(
    const std::string &device) const {
  if (!IsEnabled()) {
    return false;
  }
  assert(!device.empty());
  return !ProxyName::is_proxy_device_name(device) &&
         ProxyName::is_proxyable_device(device);
}

void ProxyComputationClient::ReleaseXrtData(const std::string &device,
                                            int64 handle) {
  if (ProxyName::is_proxy_device_name(device)) {
    if (data_mapper_->HasMapping(device, handle)) {
      data_mapper_->FreeMapping(device, handle, false);
    } else {
      /// This data is natively on the proxy device
      client_manager_.GetComputationClient(device)->ReleaseDataByHandle(device,
                                                                        handle);
    }
  } else {
    /// is it a torch_xla device?
    assert(!device.empty());
    assert(!always_use_proxy);

    Super::ReleaseXrtData(device, handle);

    if (ShouldCloneDataForDevice(device)) {
      data_mapper_->FreeMapping(device, handle, true);
    }
  }
}

/**
 *  _____        _           __  __                                         _
 * |  __ \      | |         |  \/  |                                       | |
 * | |  | | __ _| |_  __ _  | \  / | ___ __   __ ___  _ __ ___   ___  _ __ | |_
 * | |  | |/ _` | __|/ _` | | |\/| |/ _ \\ \ / // _ \| '_ ` _ \ / _ \| '_ \| __|
 * | |__| | (_| | |_| (_| | | |  | | (_) |\ V /|  __/| | | | | |  __/| | | | |_
 * |_____/ \__,_|\__|\__,_| |_|  |_|\___/  \_/  \___||_| |_| |_|\___||_| |_|\__|
 *
 *
 */
/// Transfers local tensor values to the TPU servers and fetches the handles.
std::vector<ComputationClient::DataPtr>
ProxyComputationClient::TransferToServer(
    absl::Span<const TensorSource> tensors) {
  if (verbose_transfer || verbose) {
    torch_xla::ColorScope clr(torch_xla::Color::FG_YELLOW);
    std::cout << getpid() << " ProxyComputationClient::TransferToServer( ";
    size_t i = 0;
    for (const TensorSource &t : tensors) {
      if (i++) {
        std::cout << ", ";
      }
      std::cout << t.shape << "@" << t.device;
    }
    std::cout << ")" << std::endl;
  }
  auto results = split_types<std::vector<ComputationClient::DataPtr>>(
      tensors,
      [this](const TensorSource &tensor_source) {
        /// won't work, actually... need proper device set on data ptr
        /// return UseProxyForDevice(tensor_source.device);
        return IsEnabled() &&
               ProxyName::is_proxy_device_name(tensor_source.device);
      },
      [this](const std::vector<TensorSource> &local_tensor_sources) {
        //
        // PROXY
        //
        /// Handle multiple destination devices in the list
        std::unordered_map<std::string, std::vector<std::size_t>>
            device_to_indexes;
        for (std::size_t index = 0, n = local_tensor_sources.size(); index < n;
             ++index) {
          const TensorSource &tensor_source = local_tensor_sources[index];
          device_to_indexes[tensor_source.device].emplace_back(index);
        }

        /// TODO: Parallelize with MultiWait
        std::vector<ComputationClient::DataPtr> local_results(
            local_tensor_sources.size());
        for (const auto &item : device_to_indexes) {
          const std::string &device = item.first;
          const std::vector<std::size_t> &indexes = item.second;
          /// It would be better if we could do this by reference
          /// rather than a copy, but the API takes what it takes.
          std::vector<TensorSource> tensors;
          for (std::size_t index : indexes) {
            tensors.emplace_back(local_tensor_sources[index]);
          }
          std::vector<ComputationClient::DataPtr> res =
              client_manager_.GetComputationClient(device)->TransferToServer(
                  tensors);
          /// Put back in their proper result index position
          for (std::size_t i = 0, n = tensors.size(); i < n; ++i) {
            local_results[indexes[i]] = res[i];
          }
        }
        return local_results;
      },
      [this](const std::vector<TensorSource> &local_tensor_sources) {
        //
        /// XRT
        //
        std::vector<ComputationClient::DataPtr> local_results =
            Super::TransferToServer(local_tensor_sources);
        //
        //
        /// Clone all data to proxy device
        //
        //
        if (IsEnabled()) {
          /// Temporary until re-enable device-switching in
          /// torch_xla/csrc/tensor.cpp and write the "move data" code
          std::vector<TensorSource> clone_ts;
          clone_ts.reserve(local_tensor_sources.size());
          std::vector<ComputationClient::DataPtr> original_results;
          original_results.reserve(local_tensor_sources.size());
          for (size_t i = 0; i < local_tensor_sources.size(); ++i) {
            const TensorSource &ts = local_tensor_sources[i];
            if (ShouldCloneDataForDevice(ts.device)) {
              clone_ts.emplace_back(TensorSource(
                  ts.shape, ProxyName::proxy_device_name(ts.device),
                  ts.populate_fn));
              original_results.emplace_back(local_results[i]);
            }
          }
          std::vector<ComputationClient::DataPtr> cloned_results =
              TransferToServer(clone_ts);
          assert(original_results.size() == cloned_results.size());
          for (size_t i = 0; i < cloned_results.size(); ++i) {
            ComputationClient::DataPtr orig = original_results[i];
            ComputationClient::DataPtr cloned = cloned_results[i];
            data_mapper_->AddMapping(orig->device(), orig->GetOpaqueHandle(),
                                     cloned);
          }
        }
        return local_results;
      });
  return results;
}

/// Reads the tensor literal values stored at TPU server sites, behind the
/// supplied handles.
std::vector<Literal> ProxyComputationClient::TransferFromServer(
    absl::Span<const DataPtr> handles) {
  std::vector<DataPtr> all_handles(handles.begin(), handles.end());
  std::vector<Literal> results = split_types<std::vector<Literal>>(
      all_handles,
      [this](const DataPtr &data_ptr) {
        return IsEnabled() &&
               ProxyName::is_proxy_device_name(data_ptr->device());
      },
      [this](std::vector<DataPtr> &proxy_handles) {
        //
        // PROXY
        //
        // Handle multiple source devices in the list
        std::unordered_map<std::string, std::vector<std::size_t>>
            device_to_indexes;
        for (std::size_t index = 0, n = proxy_handles.size(); index < n;
             ++index) {
          DataPtr &data_ptr = proxy_handles[index];
          device_to_indexes[data_ptr->device()].emplace_back(index);
        }

        /// TODO: Parallelize with MultiWait
        std::vector<xla::Literal> local_results(proxy_handles.size());
        for (const auto &item : device_to_indexes) {
          const std::string &device = item.first;
          const std::vector<std::size_t> &indexes = item.second;
          /// It would be better if we could do this by reference
          /// rather than a copy, but the API takes what it takes.
          std::vector<DataPtr> tensors;
          for (std::size_t index : indexes) {
            tensors.emplace_back(proxy_handles[index]);
          }
          std::vector<Literal> res =
              client_manager_.GetComputationClient(device)->TransferFromServer(
                  tensors);
          /// Put back in their proper result index position
          for (std::size_t i = 0, n = tensors.size(); i < n; ++i) {
            local_results[indexes[i]] = std::move(res[i]);
          }
        }
        return local_results;
      },
      [this](std::vector<DataPtr> &other_handles) {
        /// CPU or other (false)
        assert(!always_use_proxy);
        if (verbose || verbose_transfer || verbose_pull) {
          for (const DataPtr &d : other_handles) {
            torch_xla::ColorScope clr(torch_xla::Color::FG_RED);
            std::cout << getpid()
                      << " *XRT* ProxyComputationClient::TransferFromServer() "
                      << " handle = " << d->GetOpaqueHandle()
                      << ", shape = " << d->shape() << "@" << d->device()
                      << ENDL;
          }
        }

        return Super::TransferFromServer(other_handles);
      });
  return results;
}

std::vector<ComputationClient::DataPtr>
ProxyComputationClient::NormalizeDataToDevice(absl::Span<const DataPtr> tensors,
                                              const std::string &device,
                                              bool in_place) {
  //
  /// Split by whether to move
  //
  auto results = split_types<std::vector<ComputationClient::DataPtr>>(
      tensors,
      /*test=*/
      [&device](const ComputationClient::DataPtr &tensor) {
        return tensor->device() == device;
      },
      [](std::vector<ComputationClient::DataPtr> &local_tensors) {
        return std::move(local_tensors);
      },
      /*different device=*/
      [this, in_place,
       &device](std::vector<ComputationClient::DataPtr> &local_tensors) {
        //
        /// Split again by move direction direction
        //
        auto move_results = split_types<
            std::vector<ComputationClient::DataPtr>>(
            local_tensors,
            [](const ComputationClient::DataPtr &data_ptr) {
              return ProxyName::is_proxy_device_name(data_ptr->device());
            },
            [this, in_place](
                std::vector<ComputationClient::DataPtr> &local_move_tensors) {
              //
              /// PROXY -> XRT
              //
              std::vector<Literal> literals =
                  TransferFromServer(local_move_tensors);
              assert(literals.size() == local_move_tensors.size());
              std::vector<TensorSource> tensor_sources;
              tensor_sources.reserve(literals.size());

              for (std::size_t i = 0; i < local_move_tensors.size(); ++i) {
                tensor_sources.emplace_back(
                    literal_to_tensor(std::move(literals[i]),
                                      ProxyName::unproxy_device_name(
                                          local_move_tensors[i]->device())));
              }

              /// Add mapping entries to map a free of the new local XRT handle
              /// to free the remote proxy handle
              std::vector<ComputationClient::DataPtr> results =
                  TransferToServer(tensor_sources);
              if (in_place) {
                /// modify the data pointers in-place
                std::size_t index = 0;
                for (ComputationClient::DataPtr transferred_tensor : results) {
                  /// in-place
                  results[index] = local_move_tensors[index];
                  results[index]->Assign(*transferred_tensor);
                  ++index;
                }
                return results;
              } else {
                for (size_t i = 0; i < local_move_tensors.size(); ++i) {
                  data_mapper_->AddMapping(results[i]->device(),
                                           results[i]->GetOpaqueHandle(),
                                           local_move_tensors[i]);
                }

                return results;
              }
            },
            /*same device=*/
            [this](
                std::vector<ComputationClient::DataPtr> &local_move_tensors) {
              std::vector<ComputationClient::DataPtr> results;
              results.reserve(local_move_tensors.size());
              //
              /// XRT -> PROXY
              //
              for (DataPtr &argument : local_move_tensors) {
                const std::string xrt_device = argument->device();
                assert(!ProxyName::is_proxy_device_name(xrt_device));
                const std::string proxy_device =
                    ProxyName::proxy_device_name(xrt_device);
                if (data_mapper_->HasMapping(argument->device(),
                                             argument->GetOpaqueHandle())) {
                  DataPtr mapped_argument = data_mapper_->GetMapping(
                      argument->device(), argument->GetOpaqueHandle());
                  if (mapped_argument) {
                    argument = mapped_argument;
                  } else {
                    /// TODO: use split_types() to do in batches
                    std::vector<DataPtr> arguments_to_move{argument};
                    std::vector<DataPtr> moved_arguments =
                        MoveDataBetweenDevices(arguments_to_move, proxy_device,
                                               /*release_from_source=*/false);
                    if (verbose) {
                      std::cout << "Moved data for argument: "
                                << argument->GetOpaqueHandle() << " @"
                                << argument->device() << " ==> "
                                << moved_arguments[0]->GetOpaqueHandle() << " @"
                                << moved_arguments[0]->device() << std::endl;
                    }
                    argument = moved_arguments[0];
                  }
                } else {
                  torch_xla::ColorScope red(torch_xla::Color::FG_RED);
                  if (verbose) {
                    std::cout << "\t*** No mapping for argument handle:"
                              << argument->GetOpaqueHandle() << " @"
                              << argument->device() << std::endl;
                  }
                }
                if (verbose) {
                  std::cout << "\t-> effective argument handle: "
                            << argument->GetOpaqueHandle() << " @"
                            << argument->device()
                            << " shape = " << argument->shape() << std::endl;
                }
                results.emplace_back(argument);
              }
              assert(results.size() == local_move_tensors.size());
              std::for_each(results.begin(), results.end(), [](const auto &t) {
                assert(ProxyName::is_proxy_device_name(t->device()));
              });
              return results;
            });
        return move_results;
      });
  return results;
}

/**
 * @brief Potentially move data between devices
 */
std::vector<ComputationClient::DataPtr>
ProxyComputationClient::MoveDataBetweenDevices(
    const std::vector<ComputationClient::DataPtr> &source_data,
    const std::string &to_device, bool release_from_source) {
  auto results = split_types<std::vector<ComputationClient::DataPtr>>(
      source_data,
      /*is destination deviec same as data's current device?*/
      [&to_device](const ComputationClient::DataPtr &data_ptr) {
        return data_ptr->device() != to_device;
      },
      /*not on destination device=*/
      [this, &to_device, release_from_source](
          const std::vector<ComputationClient::DataPtr> &local_source_data) {
        /// Copy data from the other device locally and then
        /// push it to the specified device as well as add the
        /// mapping.
        /// Generally, one of these devices will be "local", so
        /// the dual move should be almost trivial.
        /// That being said, "normalization" operatioins are generally
        /// not at a performance-sensitive point, such as a one-time
        /// device switch
        std::vector<Literal> literals = TransferFromServer(local_source_data);

        std::vector<ComputationClient::TensorSource> intermediate_data;
        std::vector<ComputationClient::DataPtr> local_results;

        intermediate_data.reserve(literals.size());

        for (std::size_t i = 0, n = literals.size(); i < n; ++i) {
          Literal &literal = literals[i];
          intermediate_data.emplace_back(
              literal_to_tensor(std::move(literal), to_device));
        }

        if (ProxyName::is_proxy_device_name(to_device)) {
          /// Send to the proxy device
          local_results =
              client_manager_.GetComputationClient(to_device)->TransferToServer(
                  intermediate_data);
        } else {
          /// Send to the "local" device
          local_results = TransferToServer(intermediate_data);
        }
        /// Either release the "old" handle or add handle mapping
        /// between the two devices which both now have the data
        for (std::size_t index = 0, n = local_results.size(); index < n;
             ++index) {
          const ComputationClient::DataPtr &old_data = local_source_data[index];
          if (release_from_source) {
            ReleaseXrtData(old_data->device(), old_data->GetOpaqueHandle());
          } else {
            data_mapper_->AddMapping(old_data->device(),
                                     old_data->GetOpaqueHandle(),
                                     local_results[index]);
          }
        }
        return local_results;
      },
      /*already on destination device=*/
      [](const std::vector<ComputationClient::DataPtr> &local_source_data) {
        /// Data is already on the desired device, so do nothing
        return std::move(local_source_data);
      });
  return results;
}
/**
 *   _____                       _  _
 *  / ____|                     (_)| |
 * | |      ___  _ __ ___  _ __  _ | | ___
 * | |     / _ \| '_ ` _ \| '_ \| || |/ _ \
 * | |____| (_) | | | | | | |_) | || |  __/
 *  \_____|\___/|_| |_| |_| .__/|_||_|\___|
 *                        | |
 *                        |_|
 *
 * Compiles a set of computations.
 */
std::vector<ComputationClient::ComputationPtr> ProxyComputationClient::Compile(
    std::vector<CompileInstance> instances) {
  //
  /// TODO: ComputationPtr to return have modified HloModule and
  ///       call Super with it (no proxy) on compile failure
  //
  auto results = split_types<std::vector<ComputationClient::ComputationPtr>>(
      instances,
      [this /*, &compilation_device*/](
          const CompileInstance &instance) -> bool {
        if (always_use_proxy) {
          return true;
        }
        const std::string proxy_device =
            get_proxy_device(instance.computation.proto());
        if (proxy_device.empty()) {
          return false;
        }
        assert(proxy_device == instance.compilation_device);
        return true;
      },
      [this](std::vector<CompileInstance> &instances) {
        /// PROXY (true)
        /// TODO: Parallelize with MultiWait (previously wasn't a use-case)
        std::vector<ComputationClient::ComputationPtr> local_results;
        local_results.reserve(instances.size());
        for (auto &instance : instances) {
          const std::string compilation_device =
              ProxyName::proxy_device_name(instance.compilation_device);
          std::vector<CompileInstance> single_instance;
          single_instance.emplace_back(std::move(instance));
          std::vector<ComputationClient::ComputationPtr> inner_results =
              client_manager_.GetComputationClient(compilation_device)
                  ->Compile(std::move(single_instance));
          for (auto &executable : inner_results) {
            if (executable->execution_handle()) {
              AddProxyExecutable(executable->execution_handle());
            }
            local_results.emplace_back(std::move(executable));
          }
        }
        return local_results;
      },
      [this](std::vector<CompileInstance> &instances) {
        assert(!always_use_proxy);
        /// CPU or other (false)
        if (verbose) {
          std::cout << "Delegating compile" << std::endl;
        }
        return Super::Compile(std::move(instances));
      });
  assert(results.size() == instances.size());
  return results;
}

/**
 *  ______                        _   _
 * |  ____|                      | | (_)
 * | |__   __  __ ___   ___ _   _| |_ _  ___  _ __
 * |  __|  \ \/ // _ \ / __| | | | __| |/ _ \| '_ \
 * | |____  >  <|  __/| (__| |_| | |_| | (_) | | | |
 * |______|/_/\_\\___| \___|\__,_|\__|_|\___/|_| |_|
 *
 *
 */
bool ProxyComputationClient::IsProxyExecutable(
    uint64_t executable_handle) const {
  if (!executable_handle) {
    return false;
  }
  std::lock_guard<std::mutex> lk(proxy_executable_set_mtx_);
  return proxy_executable_set_.count(executable_handle) != 0;
}

void ProxyComputationClient::AddProxyExecutable(uint64_t executable_handle) {
  assert(executable_handle);
  if (executable_handle) {
    std::lock_guard<std::mutex> lk(proxy_executable_set_mtx_);
    proxy_executable_set_.insert(executable_handle);
  }
}

/// Executes computation with arguments and returns the result.
/// The passed device must match the common device of the arguments Data.
/// If options.explode_tuple is true, the output tuple will be decomposed into
/// its single elements.
std::vector<ComputationClient::DataPtr>
ProxyComputationClient::ExecuteComputation(
    const Computation &computation, absl::Span<const DataPtr> arguments,
    const std::string &device, const ExecuteComputationOptions &options) {
  if (verbose) {
    auto comp = dynamic_cast<const XrtComputation *>(&computation);
    if (comp) {
      std::cout << "ProxyComputationClient::ExecuteComputation( XRT ): HANDLE="
                << comp->get_handle() << std::endl;
    }
  }

  /// Probably this needs to be worked out for replication once
  /// we can use XrtComputationClient as a proxy target (see notes)
  const std::string device1 = device;
  const std::string device2 =
      get_proxy_device(computation.computation().proto());
  std::string effective_device;
  if (device1.empty() && !device2.empty()) {
    effective_device = device2;
  } else if (!device1.empty() && device2.empty()) {
    effective_device = device1;
  } else {
    assert(device1 == device2);  /// what's this use-case?
    effective_device =
        device2;  /// prefer the proxy effective_device if it's specified
  }

  if (IsProxyExecutable(computation.execution_handle())) {
    /// Execute on thee proxy device
    effective_device = ProxyName::proxy_device_name(device);
    /// Make sure any arguments we pass are on the target device
    std::vector<ComputationClient::DataPtr> args =
        NormalizeDataToDevice(arguments, effective_device, false);
    return client_manager_.GetComputationClient(effective_device)
        ->ExecuteComputation(computation, std::move(args), effective_device,
                             options);
  } else {
    /// This is the normal XRT path
    assert(!always_use_proxy);
    std::vector<DataPtr> new_args =
        NormalizeDataToDevice(arguments, effective_device, true /*false*/);

    if (verbose) {
      torch_xla::ColorScope clr(torch_xla::Color::FG_RED);
      std::cout << "Local Execution handle: " << computation.execution_handle()
                << " " << computation.program_shape().ToString() << std::endl;
    }

    std::vector<ComputationClient::DataPtr> results = Super::ExecuteComputation(
        computation, new_args, effective_device, options);

    if (verbose) {
      for (const auto &d : results) {
        torch_xla::ColorScope clr(torch_xla::Color::FG_CYAN);
        std::cout << "LOCAL Execution result data: " << d->GetOpaqueHandle()
                  << " @ " << d->device()
                  << ", shape = " << d->shape().ToString() << std::endl;
      }
    }

    //
    /// Don't move the data, but make a weak mapping of the
    /// result handles so that if we see them again, we'll know that
    /// we can get their data from the non-proxy device
    //
    std::vector<ComputationClient::DataPtr> cloned_results;
    cloned_results.reserve(results.size());
    for (ComputationClient::DataPtr &data_ptr : results) {
      if (!ProxyName::is_proxy_device_name(data_ptr->device())) {
        data_mapper_->AddWeakMapping(data_ptr->device(),
                                     data_ptr->GetOpaqueHandle());
      }
    }
    if (verbose) {
      for (const auto &d : cloned_results) {
        torch_xla::ColorScope clr(torch_xla::Color::FG_MAGENTA);
        std::cout << "WEAK MAPPED FROM Execution result data: "
                  << d->GetOpaqueHandle() << " @ " << d->device()
                  << ", shape = " << d->shape().ToString() << std::endl;
      }
    }

    return results;
  }
}

std::vector<std::vector<ComputationClient::DataPtr>>
ProxyComputationClient::ExecuteParallel(
    absl::Span<const ComputationClient::Computation *const> computations,
    const std::vector<std::vector<DataPtr>> &arguments,
    absl::Span<const std::string> devices,
    const ExecuteParallelOptions &options) {
  struct ParallelInfo {
    const Computation *const computation;
    const std::vector<DataPtr> *args;
    const std::string device;
  };
  std::vector<ParallelInfo> parallel_infos;
  for (std::size_t i = 0, n = devices.size(); i < n; ++i) {
    parallel_infos.emplace_back(ParallelInfo{.computation = computations[i],
                                             .args = &arguments[i],
                                             .device = devices[i]});
  }
  auto results =
      split_types<std::vector<std::vector<ComputationClient::DataPtr>>>(
          parallel_infos,
          [this](const ParallelInfo &parallel_info) -> bool {
            if (always_use_proxy) {
              return true;
            }
            return IsProxyExecutable(
                parallel_info.computation->execution_handle());
          },
          [this, &options](const std::vector<ParallelInfo> &parallel_infos) {
            /// PROXY
            std::vector<Computation const *> local_computations;
            std::vector<std::vector<DataPtr>> local_arguments;
            std::vector<std::string> local_devices;
            for (auto &pinfo : parallel_infos) {
              local_computations.emplace_back(pinfo.computation);
              local_arguments.emplace_back(*pinfo.args);
              local_devices.emplace_back(pinfo.device);
            }
            // Unused use-case so far.
            // Will need to determine how this device split goes.
            // Use device index 0 or make a single list for each?
            // Or split up on device for which the computation has been
            // registered a proxy computation?
            return client_manager_.GetComputationClient(local_devices[0])
                ->ExecuteParallel(local_computations, local_arguments,
                                  local_devices, options);
          },
          [this, &options](const std::vector<ParallelInfo> &parallel_infos) {
            /// XRT
            std::vector<Computation const *> local_computations;
            std::vector<std::vector<DataPtr>> local_arguments;
            std::vector<std::string> local_devices;
            for (auto &pinfo : parallel_infos) {
              local_computations.emplace_back(pinfo.computation);
              local_arguments.emplace_back(*pinfo.args);
              local_devices.emplace_back(pinfo.device);
            }
            return Super::ExecuteParallel(local_computations, local_arguments,
                                          local_devices, options);
          });
  assert(results.size() == computations.size());
  return results;
}

std::vector<std::vector<ComputationClient::DataPtr>>
ProxyComputationClient::ExecuteReplicated(
    const Computation &computation,
    const std::vector<std::vector<DataPtr>> &arguments,
    absl::Span<const std::string> devices,
    const ExecuteReplicatedOptions &options) {
  struct ReplicatedInfo {
    const std::vector<DataPtr> *args;
    const std::string device;
  };
  std::vector<ReplicatedInfo> replicated_infos;
  for (std::size_t i = 0, n = devices.size(); i < n; ++i) {
    replicated_infos.emplace_back(
        ReplicatedInfo{.args = &arguments[i], .device = devices[i]});
  }
  auto results =
      split_types<std::vector<std::vector<ComputationClient::DataPtr>>>(
          replicated_infos,
          [this, &computation](const ReplicatedInfo &replicated_info) -> bool {
            if (always_use_proxy) {
              return true;
            }
            const bool proxy =
                IsProxyExecutable(computation.execution_handle());
            assert(proxy ==
                   ProxyName::is_proxy_device_name(replicated_info.device));
            return proxy;
          },
          [this, &computation,
           &options](const std::vector<ReplicatedInfo> &replicated_infos) {
            /// PROXY
            std::vector<std::vector<DataPtr>> local_arguments;
            std::vector<std::string> local_devices;
            for (auto &pinfo : replicated_infos) {
              local_arguments.emplace_back(*pinfo.args);
              local_devices.emplace_back(pinfo.device);
            }
            // Unused use-case so far.
            // Will need to determine how this device split goes.
            // Use device index 0 or make a single list for each?
            // Or split up on device for which the computation has been
            // registered a proxy computation?
            return client_manager_.GetComputationClient(local_devices[0])
                ->ExecuteReplicated(computation, local_arguments, local_devices,
                                    options);
          },
          [this, &computation,
           &options](const std::vector<ReplicatedInfo> &replicated_infos) {
            /// XRT
            std::vector<std::vector<DataPtr>> local_arguments;
            std::vector<std::string> local_devices;
            for (auto &pinfo : replicated_infos) {
              local_arguments.emplace_back(*pinfo.args);
              local_devices.emplace_back(pinfo.device);
            }
            return Super::ExecuteReplicated(computation, local_arguments,
                                            local_devices, options);
          });
  assert(results.size() == devices.size());
  return results;
}

std::vector<ComputationClient::DataPtr> ProxyComputationClient::ExecuteChained(
    absl::Span<const ExecuteChainedOp> ops, const std::string &device) {
  // Have never run this use-case, so I doubt this works for proxy :)
  if (IsEnabled() && ProxyName::is_proxy_device_name(device)) {
    return client_manager_.GetComputationClient(device)->ExecuteChained(ops,
                                                                        device);
  } else {
    return Super::ExecuteChained(ops, device);
  }
}

/**
 *  _______                _
 * |__   __|              | |
 *    | | ___  _ __   ___ | | ___   __ _ _   _
 *    | |/ _ \| '_ \ / _ \| |/ _ \ / _` | | | |
 *    | | (_) | |_) | (_) | | (_) | (_| | |_| |
 *    |_|\___/| .__/ \___/|_|\___/ \__, |\__, |
 *            | |                   __/ | __/ |
 *            |_|                  |___/ |___/
 */
///
/// Override topology to some non-tpu, yet distributed device
/// This is deprecated in favor of using the proper TPU config OP
///
tensorflow::tpu::TopologyProto
ProxyComputationClient::InitializeAndFetchTopology(
    const std::string &job, int task_no, const std::string &worker_host_port,
    const tensorflow::ConfigProto &config) {
  if (verbose_topology) {
    std::cout << "InitializeAndFetchTopology: job=" << job
              << ", task_no=" << task_no
              << ", worker_host_port=" << worker_host_port
              << ", config=" << msg_to_json(config) << std::endl;
  }
  const int proxy_num_devices = sys_util::GetEnvInt("WSE_NUM_DEVICES", 0);
  if (!proxy_num_devices && (ComputationClientManager::Empty() ||
                             !sys_util::GetEnvBool("WSE_TPU_MODE", false))) {
    if (verbose_topology) {
      std::cout << "** Falling back to normal InitializeAndFetchTopology()"
                << std::endl;
    }
    return Super::InitializeAndFetchTopology(job, task_no, worker_host_port,
                                             config);
  }
  return InitializeAndFetchTopologyLocal(job, task_no, worker_host_port,
                                         config);
}

}  // namespace xla
