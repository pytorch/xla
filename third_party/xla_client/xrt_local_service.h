#ifndef XLA_CLIENT_XRT_LOCAL_SERVICE_H_
#define XLA_CLIENT_XRT_LOCAL_SERVICE_H_

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/debug_macros.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"

namespace xla {

// A TF server running on a local interface.
class XrtLocalService {
 public:
  // The cluster_spec has format:
  //   CLUSTER_SPEC = JOB,...
  //   JOB          = NAME|ADDRESS_LIST
  //   NAME         = The name of the job
  //   ADDRESS_LIST = HOST:PORT;...
  //   HOST         = Hostname or IP address
  //   PORT         = Port number
  //
  // The job_name must match one of the job names in the cluster_spec, and
  // represents this job.
  // The task_index must be within the range of the ADDRESS_LIST of the current
  // job in the cluster_spec.
  XrtLocalService(const std::string& cluster_spec, const std::string& job_name,
                  int task_index);

  // Starts the service.
  void Start();

 private:
  std::unique_ptr<tensorflow::ServerInterface> server_;
};

}  // namespace xla

#endif  // XLA_CLIENT_XRT_LOCAL_SERVICE_H_
