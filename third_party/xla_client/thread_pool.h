#ifndef TENSORFLOW_COMPILER_XLA_XLA_CLIENT_THREAD_POOL_H_
#define TENSORFLOW_COMPILER_XLA_XLA_CLIENT_THREAD_POOL_H_

#include <functional>

namespace xla {
namespace xla_env {

// Schedules a closure to be run. The closure should not block.
void ScheduleClosure(std::function<void()> closure);

// Schedules a closure which might wait for IO or other events/conditions.
void ScheduleIoClosure(std::function<void()> closure);

}  // namespace xla_env
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_XLA_CLIENT_THREAD_POOL_H_
