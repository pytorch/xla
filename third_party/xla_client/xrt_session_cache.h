#ifndef TENSORFLOW_COMPILER_XLA_RPC_XRT_SESSION_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_RPC_XRT_SESSION_CACHE_H_

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <utility>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/xrt_session.h"

namespace xla {

class XrtSessionCache {
 public:
  class Ref {
   public:
    Ref(XrtSessionCache* cache, std::shared_ptr<XrtSession> session)
        : cache_(cache), session_(std::move(session)) {}

    Ref(Ref&& ref) { MoveFrom(std::move(ref)); }

    Ref(const Ref&) = delete;

    ~Ref() { Detach(); }

    Ref& operator=(Ref&& rhs) {
      if (&rhs != this) {
        MoveFrom(std::move(rhs));
      }
      return *this;
    }

    Ref& operator=(const Ref&) = delete;

    XrtSession* operator->() const { return get(); }

    XrtSession* get() const { return session_.get(); }

   private:
    void MoveFrom(Ref&& rhs) {
      Detach();
      cache_ = rhs.cache_;
      rhs.cache_ = nullptr;
      session_ = std::move(rhs.session_);
    }

    void Detach() {
      if (cache_ != nullptr) {
        cache_->AddSession(std::move(session_));
        cache_ = nullptr;
      }
    }

    XrtSessionCache* cache_ = nullptr;
    std::shared_ptr<XrtSession> session_;
  };

  using SessionMap = std::map<string, Ref>;

  Ref GetSession(const string& target);

  XrtSession* GetSession(const string& target, SessionMap* session_map);

  void AddSession(std::shared_ptr<XrtSession> session);

 private:
  std::shared_ptr<XrtSession> CreateSession(const string& target) const;

  std::mutex lock_;
  std::map<string, std::deque<std::shared_ptr<XrtSession>>> session_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_XRT_SESSION_CACHE_H_
