#ifndef XLA_CLIENT_PROFILER_H_
#define XLA_CLIENT_PROFILER_H_

#include <memory>

namespace xla {
namespace profiler {

class ProfilerServer {
  struct Impl;

 public:
  ProfilerServer();
  ~ProfilerServer();
  void Start(int port);

 private:
  std::unique_ptr<Impl> impl_;
};

class ProfilerSession {
  struct Impl;

 public:
  ProfilerSession();
  ~ProfilerSession();
  // void Start(int port);
  tensorflow::ProfileOptions DefaultPythonProfileOptions();
  static std::unique_ptr<ProfilerSession> Create(
    const tensorflow::ProfileOptions& options);

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_CLIENT_PROFILER_H_
