#ifndef XLA_TORCH_XLA_CSRC_COMMON_SINGLETON_H_
#define XLA_TORCH_XLA_CSRC_COMMON_SINGLETON_H_

#include <pthread.h>

#include <atomic>
#include <list>
#include <memory>
#include <mutex>

#include "torch_xla/csrc/common/base.h"

namespace lynx {

/// @brief This class abstracts the Singleton design pattern. It requires the
/// real singleton
///        class to derive from this class, such as: class Derived : public
///        Singleton<Derive>. Furthermore, the Derived class should have a
///        friend class Singleton<Derived> to allow Singleton to call the
///        private constructor.
///
/// @tparam T The derived class.
template <class T>
class Singleton {
 public:
  /// @brief Get the global singleton instance. Only the first call constructs
  /// the object.
  ///
  /// @return T*
  static T *GetInstance();

  /// @brief This is needed in ASAN. For a static singleton variable, we also do
  /// not delete it,
  ///        but the system will delete it before program exiting.
  ///        In ASAN UT case, this function should be called to avoid ASAN
  ///        failure.
  ///
  static void Destroy();

 protected:
  /// @brief If the derived Singleton class does not define this function, `new
  /// T` will be used.
  ///        Otherwise, the ManualCreate function defined in the derived class
  ///        will be used.
  ///
  /// @return T*
  static T *ManualCreate();

  /// @brief Whether this Singleton object is managed by SingletonManager.
  ///        By default, it should be put to it.
  ///
  /// @return true All Singleton objects except the SingletonManager should be
  /// true.
  /// @return false Only SingletonManager object is false.
  static bool ShouldPutToManager() { return true; }

 private:
  // The singleton object.
  static std::atomic<T *> pointer_;
  // The mutex used during object creation time.
  static std::mutex mutex_;

 protected:
  Singleton() = default;
  DISALLOW_COPY_AND_MOVE(Singleton<T>);
};

template <class T>
std::atomic<T *> Singleton<T>::pointer_{nullptr};
template <class T>
std::mutex Singleton<T>::mutex_;

// Foward declaration.
class SingletonResourceHolderBase;

/// @brief Used to manage all singleton objects. We can destroy all singleton
/// objects by DestroyAll.
///
class SingletonManager : public Singleton<SingletonManager> {
 public:
  /// @brief Put an object with class type T.
  ///
  /// @tparam T The singleton object type.
  /// @param ptr The singleton object.
  template <typename T>
  void Put(T *ptr);

  /// @brief Clear all singleton objects including SingletonManager itself.
  ///
  static void DestroyAll() {
    SingletonManager::GetInstance()->ClearResources();
    SingletonManager::Destroy();
  }

  /// @brief SingletonManager should not be added to SingletonManager.
  ///
  static bool ShouldPutToManager() { return false; }

 private:
  friend class Singleton<SingletonManager>;
  std::mutex mutex_;
  SingletonManager() = default;
  void ClearResources() { resources_.clear(); }
  std::list<std::unique_ptr<SingletonResourceHolderBase>> resources_;
};

template <class T>
T *Singleton<T>::GetInstance() {
  if (LIKELY(pointer_.load(std::memory_order_acquire) != nullptr)) {
    return pointer_;
  }
  T *ptr = nullptr;
  {
    std::lock_guard<std::mutex> guard(mutex_);
    if (pointer_.load(std::memory_order_relaxed) != nullptr) {
      return pointer_;
    }
    // If there is no T::ManualCreate defined, Singleton<T>::ManualCreate will
    // be used.
    pointer_.store(T::ManualCreate(), std::memory_order_release);
    ptr = pointer_.load(std::memory_order_relaxed);
  }
  if (T::ShouldPutToManager()) {
    SingletonManager *manager = SingletonManager::GetInstance();
    manager->Put(ptr);
  }
  return ptr;
}

template <class T>
void Singleton<T>::Destroy() {
  if (pointer_.load(std::memory_order_acquire) != nullptr) {
    T *pointer = pointer_.exchange(nullptr, std::memory_order_acq_rel);
    if (pointer != nullptr) {
      delete pointer;
    }
  }
}

template <class T>
T *Singleton<T>::ManualCreate() {
  // This function can be overwritten by writing a T::ManualCreate in the
  // derived class T.
  return new T;
}

class SingletonResourceHolderBase {
 public:
  virtual ~SingletonResourceHolderBase() = default;
};

template <typename T>
class SingletonResourceHolder : public SingletonResourceHolderBase {
 public:
  /// @brief The deleter used in std::unique_ptr.
  ///        Since the resource is a Singleton object, we use
  ///        Singleton<T>::Destroy() to delete it.
  ///
  struct Deleter {
    void operator()(T *) { Singleton<T>::Destroy(); }
  };
  explicit SingletonResourceHolder(T *ptr) : resource_(ptr) {}
  ~SingletonResourceHolder() override = default;

 private:
  std::unique_ptr<T, Deleter> resource_;
};

template <typename T>
void SingletonManager::Put(T *ptr) {
  resources_.emplace_back(std::make_unique<SingletonResourceHolder<T>>(ptr));
}

}  // namespace lynx

#endif