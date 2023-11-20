#ifndef XLA_CLIENT_THREAD_POOL_H_
#define XLA_CLIENT_THREAD_POOL_H_

#include <functional>

namespace torch_xla {
namespace thread {

// Schedules a closure to be run. The closure should not block waiting for other
// events.
void Schedule(std::function<void()> fn);

}  // namespace thread
}  // namespace torch_xla

#endif  // XLA_CLIENT_THREAD_POOL_H_
