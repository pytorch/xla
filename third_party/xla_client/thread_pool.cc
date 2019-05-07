#include "tensorflow/compiler/xla/xla_client/thread_pool.h"

#include <thread>

namespace xla {
namespace env {

void ScheduleClosure(std::function<void()> closure) {
  std::thread t(std::move(closure));
  t.detach();
}

void ScheduleIoClosure(std::function<void()> closure) {
  std::thread t(std::move(closure));
  t.detach();
}

}  // namespace env
}  // namespace xla
