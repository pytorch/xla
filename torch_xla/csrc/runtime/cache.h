#ifndef XLA_CLIENT_CACHE_H_
#define XLA_CLIENT_CACHE_H_

#include <sys/stat.h>
#include <torch/csrc/lazy/core/metrics.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace torch_xla {
namespace runtime {
namespace util {

template <typename K, typename T, typename H = std::hash<K>,
          typename E = std::equal_to<K>>
class AbstractCache {
 public:
  using TypePtr = std::shared_ptr<T>;
  virtual TypePtr Add(K key, TypePtr object) = 0;
  virtual TypePtr Get(const K& key) = 0;
  virtual size_t GetNumInMemoryCachedGraph() const = 0;
  virtual bool Erase(const K& key) = 0;
  virtual void Clear() = 0;
};

// Generic key and object cache with LRU expiration policy. The objects of type
// T will be stored as std::shared_ptr<T> and taken and returned as such, by the
// cache API.
template <typename K, typename T, typename H = std::hash<K>,
          typename E = std::equal_to<K>>
class Cache : public AbstractCache<K, T, H, E> {
 public:
  using TypePtr = std::shared_ptr<T>;
  using Element = std::pair<K, TypePtr>;
  explicit Cache(size_t max_size) : max_size_(max_size) {}

  // Adds an object to the cache, unless it already exists. If the cache grows
  // beyond the limit set during construction, the oldest used object will be
  // removed from the cache.
  TypePtr Add(K key, TypePtr object) override {
    std::lock_guard<std::mutex> slock(lock_);
    element_list_.emplace_front(Element(std::move(key), std::move(object)));
    auto it = element_list_.begin();
    auto emplace_result = element_map_.emplace(&it->first, it);
    if (!emplace_result.second) {
      element_list_.erase(it);
      DoLRU(emplace_result.first->second);
    } else if (element_list_.size() > max_size_) {
      Element* last = &element_list_.back();
      element_map_.erase(&last->first);
      element_list_.pop_back();
    }
    return emplace_result.first->second->second;
  }

  // Retrieves the existing object if it exists. If it does, it's position in
  // the LRU list gets moved to the head of the list.
  // Returns nullptr if no object with the specified key is found within the
  // cache.
  TypePtr Get(const K& key) override {
    std::lock_guard<std::mutex> slock(lock_);
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      return nullptr;
    }
    DoLRU(it->second);
    return it->second->second;
  }

  size_t GetNumInMemoryCachedGraph() const override {
    return element_list_.size();
  }

  bool Erase(const K& key) override {
    std::lock_guard<std::mutex> slock(lock_);
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      return false;
    }
    auto lit = it->second;
    element_map_.erase(it);
    element_list_.erase(lit);
    return true;
  }

  void Clear() override {
    std::lock_guard<std::mutex> slock(lock_);
    element_map_.clear();
    element_list_.clear();
  }

 private:
  using ElementList = std::list<Element>;

  struct Hasher {
    size_t operator()(const K* key) const { return hasher(*key); }

    H hasher;
  };

  struct Equaler {
    bool operator()(const K* k1, const K* k2) const {
      return equaler(*k1, *k2);
    }

    E equaler;
  };

  using ElementMap =
      std::unordered_map<const K*, typename ElementList::iterator, Hasher,
                         Equaler>;

  void DoLRU(typename ElementList::iterator it) {
    element_list_.splice(element_list_.begin(), element_list_, it);
  }

  std::mutex lock_;
  size_t max_size_ = 0;
  ElementList element_list_;
  ElementMap element_map_;
};

// A persistent cache which serializes values to disk. This wraps a Cache
// instance, so values will only be read from disk once and subsequent reads
// will go through the wrapped Cache.
template <typename K, typename T, typename H = std::hash<K>,
          typename E = std::equal_to<K>>
class PersistentCache : public AbstractCache<K, T, H, E> {
 public:
  using TypePtr = std::shared_ptr<T>;

  explicit PersistentCache(
      int kMaxMemoryCacheSize, std::string cache_dir, bool readonly_storage,
      std::function<std::string(const TypePtr&)> serialize,
      std::function<TypePtr(const std::string&)> deserialize)
      : memory_cache_(kMaxMemoryCacheSize),
        cache_dir_(cache_dir),
        readonly_storage_(readonly_storage),
        serialize_(serialize),
        deserialize_(deserialize) {
    std::filesystem::create_directories(cache_dir);
  }

  // Add the value to the persistent cache. This only writes to disk if no
  // existing value is tracked to avoid unnecessary serialization overhead.
  // The value will also be tracked in memory, so subsequent Get calls will not
  // incur deserialization.
  // If the cache is readonly, nothing is written to disk.
  TypePtr Add(K key, TypePtr obj) override {
    std::lock_guard<std::mutex> slock(lock_);
    std::string path = GetPath(key);
    if (!Exists(path) && !readonly_storage_) {
      std::ofstream out(path, std::ios::binary);
      out << serialize_(obj);
    }
    return memory_cache_.Add(key, obj);
  }

  // Get the TypePtr associated with the key. This method will first check
  // if the key is tracked in memory, and if not it will check for a persisted
  // version on disk.
  TypePtr Get(const K& key) override {
    std::lock_guard<std::mutex> slock(lock_);
    TypePtr mem = memory_cache_.Get(key);
    if (mem) {
      return mem;
    }

    std::string path = GetPath(key);
    if (!Exists(path)) {
      TORCH_LAZY_COUNTER("PersistentCacheMiss", 1);
      return nullptr;
    }
    TORCH_LAZY_TIMED("PersistentCacheLoad");
    std::stringstream ss;
    std::ifstream in(path, std::ios::binary);
    ss << in.rdbuf();
    std::string serialization = ss.str();

    TypePtr val = deserialize_(serialization);
    if (!val) {
      TORCH_LAZY_COUNTER("PersistentCacheDeserializeFailure", 1);
      // Remove the serialized value from disk to allow a new value to be stored
      EraseImpl(key);
      return nullptr;
    }
    TORCH_LAZY_COUNTER("PersistentCacheHit", 1);
    // Make sure the memory_cache_ tracks the value to prevent multiple loads
    return memory_cache_.Add(key, val);
  }

  size_t GetNumInMemoryCachedGraph() const override {
    return memory_cache_.GetNumInMemoryCachedGraph();
  }

  void Clear() override {
    std::lock_guard<std::mutex> slock(lock_);
    memory_cache_.Clear();
    // Delete and recreate the cache directory on disk.
    if (!readonly_storage_) {
      std::filesystem::remove_all(cache_dir_);
      std::filesystem::create_directories(cache_dir_);
    }
  }

  bool Erase(const K& key) override {
    std::lock_guard<std::mutex> slock(lock_);
    return EraseImpl(key);
  }

  Cache<K, T, H, E>& GetMemoryCache() { return memory_cache_; }

 private:
  std::string GetPath(K key) {
    std::stringstream ss;
    ss << key;
    return cache_dir_ / ss.str();
  }

  bool Exists(std::string path) {
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
  }

  bool EraseImpl(const K& key) {
    memory_cache_.Erase(key);
    return !readonly_storage_ && std::filesystem::remove(GetPath(key));
  }

  Cache<K, T, H, E> memory_cache_;
  std::function<std::string(const TypePtr&)> serialize_;
  std::function<TypePtr(const std::string&)> deserialize_;
  std::filesystem::path cache_dir_;
  std::mutex lock_;
  // readonly_storage_ controls whether the cache will treat the persistence
  // layer as readonly. When set, operations which mutate the cache, such as
  // Erase and Add, are not written to disk, but they are still applied to the
  // in-memory cache.
  const bool readonly_storage_;
};

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_CACHE_H_
