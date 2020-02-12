#ifndef XLA_CLIENT_XRT_SESSION_CACHE_H_
#define XLA_CLIENT_XRT_SESSION_CACHE_H_

#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_client/xrt_session.h"

namespace xla {

// Caches XrtSession objects. The XrtSession objects handed out by this class
// will be at exclusive use of the caller.
class XrtSessionCache {
 public:
  // A reference to an existing XrtSession. Its destructor will return it to the
  // cache.
  class Ref {
   public:
    Ref(XrtSessionCache* cache, std::shared_ptr<XrtSession> session)
        : cache_(cache), session_(std::move(session)) {}

    Ref(Ref&& ref) { MoveFrom(std::move(ref)); }

    Ref(const Ref&) = delete;

    ~Ref() { ReturnToCache(); }

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
      ReturnToCache();
      cache_ = rhs.cache_;
      rhs.cache_ = nullptr;
      session_ = std::move(rhs.session_);
    }

    void ReturnToCache() {
      if (cache_ != nullptr) {
        cache_->AddSession(std::move(session_));
        cache_ = nullptr;
      }
    }

    XrtSessionCache* cache_ = nullptr;
    std::shared_ptr<XrtSession> session_;
  };

  // Map from session target to XrtSession reference.
  using SessionMap = std::map<std::string, Ref>;

  XrtSessionCache(tensorflow::ConfigProto config,
                  std::function<void(XrtSession*)> initfn);

  const tensorflow::ConfigProto& GetConfig() const { return config_; }

  // Retrieves a new session reference, for which the caller will have exclusive
  // access. Once the reference object is destroyed, the session will be
  // returned to the cache.
  Ref GetSession(const std::string& target);

  // Retrieves an XRT session by first checking the references already stored in
  // the session_map, and, if missing, one will be fetched from the cache and
  // added to the session_map.
  XrtSession* GetSession(const std::string& target, SessionMap* session_map);

  void AddSession(std::shared_ptr<XrtSession> session);

 private:
  std::shared_ptr<XrtSession> CreateSession(const std::string& target) const;

  tensorflow::ConfigProto config_;
  std::function<void(XrtSession*)> initfn_;
  std::mutex lock_;
  std::map<std::string, std::deque<std::shared_ptr<XrtSession>>> session_map_;
};

}  // namespace xla

#endif  // XLA_CLIENT_XRT_SESSION_CACHE_H_
