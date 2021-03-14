#include "tensorflow/compiler/xla/xla_client/proxy_client_util.h"
#include "tensorflow/compiler/xla/xla_client/proxy_computation_client.h"

#include <google/protobuf/util/json_util.h>

namespace xla {

namespace {
bool verbose_topology = false;
}  // end of anonymous namespace

static const char *env_vars[] = {"XRT_MULTI_PROCESSING_DEVICE",
                                 "XRT_LOCAL_WORKER",
                                 "XRT_TPU_CONFIG",
                                 "XRT_MESH_SERVICE_ADDRESS",
                                 "XRT_DEVICE_MAP",
                                 "XRT_WORKERS",
                                 "XRT_SHARD_LOCAL_ORDINAL",
                                 "XRT_SHARD_ORDINAL",
                                 "XRT_SHARD_WORLD_SIZE",
                                 "XRT_HOST_WORLD_SIZE",
                                 "XRT_HOST_ORDINAL",
                                 "XRT_TORCH_DIST_METHOD",
                                 "XRT_TORCH_DIST_ROOT",
                                 "XRT_MULTI_PROCESSING_DEVICE",
                                 "TPU_NUM_DEVICES",
                                 "GPU_NUM_DEVICES",
                                 "WSE_NUM_DEVICES",
                                 "CPU_NUM_DEVICES",
                                 "WSE_TPU_MODE"};

extern "C" {
extern char **environ;
}

void print_environment_config() {
  static bool print_env_config =
      xla::sys_util::GetEnvBool("XLA_PRINT_CONFIG", false);
  if (!print_env_config) {
    return;
  }
  std::stringstream ss;
  ss << "------------------" << getpid() << " PROCESS CONFIG"
     << "------------------" << std::endl;
  int i = 0;
  while (environ[i]) {
    if (strncmp(environ[i], "XRT_", 4) ==
        0 /*|| strncmp(environ[i], "XLA_", 4) == 0*/) {
      ss << "\t" << environ[i] << std::endl;
    }
    ++i;
  }
  ss << "------------------" << std::endl;
  for (std::size_t i = 0; i < sizeof(env_vars) / sizeof(env_vars[0]); ++i) {
    const char *s = getenv(env_vars[i]);
    if (s && *s) {
      ss << "\t\"" << env_vars[i] << "\" = \"" << s << "\"" << std::endl;
    }
  }
  ss << "------------------" << std::endl;
  std::cout << ss.str();
}

std::vector<std::string> split(const std::string &str, const char delim) {
  std::vector<std::string> strings;
  std::size_t start;
  std::size_t end = 0;
  while ((start = str.find_first_not_of(delim, end)) != std::string::npos) {
    end = str.find(delim, start);
    strings.push_back(str.substr(start, end - start));
  }
  return std::move(strings);
}

std::string join(const std::vector<std::string> &pieces,
                 const std::string &delimiter) {
  std::stringstream ss;
  std::size_t i = 0;
  for (const auto &s : pieces) {
    if (i++) ss << delimiter;
    ss << s;
  }
  return ss.str();
}

template <typename DEST_MSG, typename SRC_MSG_ARRAY>
const DEST_MSG *get_id(SRC_MSG_ARRAY &array, const int64_t id) {
  const int64_t total_count = array.size();
  for (int64_t i = 0; i < total_count; ++i) {
    auto &obj = array[i];
    if (obj.id() == id) {
      return &obj;
    }
  }
  return nullptr;
}

template <typename DEST_MSG, typename SRC_MSG_ARRAY>
DEST_MSG *get_mutable_id(SRC_MSG_ARRAY &array, const int64_t id) {
  const int64_t total_count = array.size();
  for (int64_t i = 0; i < total_count; ++i) {
    auto &obj = array[i];
    if (obj.id() == id) {
      return &obj;
    }
  }
  return nullptr;
}

std::string get_frontend_attribute(const xla::HloModuleProto &module,
                                   const std::string &attribute_name) {
  if (!ProxyComputationClient::IsEnabled()) return "";
  const int64_t entry_computation_id = module.entry_computation_id();
  if (entry_computation_id) {
    const auto *computation = get_id<xla::HloComputationProto>(
        module.computations(), entry_computation_id);
    const int64_t root_id = computation->root_id();
    if (root_id) {
      const auto *root_instruction = get_id<xla::HloInstructionProto>(
          computation->instructions(), root_id);
      const xla::FrontendAttributes &frontend_attributes =
          root_instruction->frontend_attributes();
      auto iter = frontend_attributes.map().find(attribute_name);
      if (iter != frontend_attributes.map().end()) {
        return iter->second;
      }
    }
  }
  return "";
}

void set_frontend_attribute(xla::HloModuleProto &module,
                            const std::string &attribute_name,
                            std::string attribute_value) {
  const int64_t entry_computation_id = module.entry_computation_id();
  if (entry_computation_id) {
    xla::HloComputationProto *computation =
        get_mutable_id<xla::HloComputationProto>(*module.mutable_computations(),
                                                 entry_computation_id);
    const int64_t root_id = computation->root_id();
    if (root_id) {
      xla::HloInstructionProto *root_instruction =
          get_mutable_id<xla::HloInstructionProto>(
              *computation->mutable_instructions(), root_id);
      xla::FrontendAttributes *frontend_attributes =
          root_instruction->mutable_frontend_attributes();
      (*frontend_attributes->mutable_map())[attribute_name] =
          std::move(attribute_value);
    }
  }
}

std::string get_proxy_device(const xla::HloModuleProto &module) {
  return get_frontend_attribute(module, "PROXY_DEVICE");
}

std::unique_ptr<xla::HloModuleProto> get_proxy_hlo_module(
    const xla::HloModuleProto &module) {
  std::unique_ptr<xla::HloModuleProto> result;
  std::string proxy_hlo_string = get_frontend_attribute(module, "PROXY_HLO");
  if (!proxy_hlo_string.empty()) {
    result = std::make_unique<xla::HloModuleProto>();
    auto status = google::protobuf::util::JsonStringToMessage(
        std::move(proxy_hlo_string), result.get());
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
  }
  return result;
}

/*
message TopologyProto {
  // The dimensions of the TPU topology, in cores. Typically, this is a 3D
  // topology [x, y, core], where the major dimensions correspond to TPU chips,
  // and the minor dimension describes the number of cores on a multicore chip.
  repeated int32 mesh_shape = 1;

  // Number of TensorFlow tasks in the cluster.
  int32 num_tasks = 2;

  // Number of TPU devices per task.
  int32 num_tpu_devices_per_task = 3;

  // A flattened rank 3 int32 array with shape
  // [num_tasks, num_tpu_devices_per_task, len(mesh_shape)].
  // `tasks` is the number of tasks in the TPU cluster, `devices` is the number
  // of TPU devices per task, and the minor dimension corresponds to a position
  // in the TPU mesh topology. Each entry [task, device, axis] gives the
  // `axis`-th coordinate in the topology of a task/device pair.
  repeated int32 device_coordinates = 4;
}
*/
tensorflow::tpu::TopologyProto InitializeAndFetchTopologyLocal(
    const std::string &job, int task_no, const std::string &worker_host_port,
    const tensorflow::ConfigProto &config) {
  //
  // TODO: Move to TPU config op
  //
  tensorflow::tpu::TopologyProto topology_proto;

  const int num_devices_per_task = 1;

  std::map<int, std::vector<std::string>> tasks;
  const tensorflow::ClusterDef &cluster_def = config.cluster_def();
  for (const tensorflow::JobDef &job_def : cluster_def.job()) {
    for (const auto &task_id_and_host_port : job_def.tasks()) {
      const int task_id = task_id_and_host_port.first;
      const std::string &host_and_port = task_id_and_host_port.second;
      tasks[task_id].emplace_back(host_and_port);
    }
  }
  if (tasks.empty()) {
    tasks[0] = {""};
  }

  const int total_nr_cores = tasks.size() * num_devices_per_task;

  topology_proto.add_mesh_shape(1);
  topology_proto.add_mesh_shape(1);
  topology_proto.add_mesh_shape(1);
  topology_proto.add_mesh_shape(total_nr_cores * 3 /* x, y, x */);

  int core_nr = 0;
  for (int task = 0; task < tasks.size(); ++task) {
    for (int task_core = 0; task_core < num_devices_per_task; ++task_core) {
      for (int i = 0; i < topology_proto.mesh_shape_size(); ++i) {
        topology_proto.add_device_coordinates(0 /*task*/);
        topology_proto.add_device_coordinates(0 /*task_core*/);
        topology_proto.add_device_coordinates(0 /*core_nr++*/);
      }
    }
  }

  topology_proto.set_num_tasks(tasks.size());

  topology_proto.set_num_tpu_devices_per_task(1 /*proxy_num_devices ? */);

  return topology_proto;
}

}  // namespace xla
