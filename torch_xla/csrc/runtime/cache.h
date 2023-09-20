#ifndef XLA_CLIENT_CACHE_H_
#define XLA_CLIENT_CACHE_H_

#include <sys/stat.h>

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

template <typename K, typename T, typename H = std::hash<K>,
          typename E = std::equal_to<K>>
class PersistentCache : public AbstractCache<K, T, H, E> {
 public:
  using TypePtr = std::shared_ptr<T>;

  explicit PersistentCache(
      int kMaxCacheSize, std::string cache_dir, bool readonly,
      std::function<void(TypePtr&, std::ostream&)> serialize,
      std::function<TypePtr(std::istream&)> deserialize)
      : memcache_(kMaxCacheSize),
        cache_dir_(cache_dir),
        readonly_(readonly),
        serialize_(serialize),
        deserialize_(deserialize) {
    std::filesystem::create_directories(cache_dir);
  }

  TypePtr Add(K key, TypePtr obj) override {
    std::string path = GetPath(key);
    if (!Exists(path) && !readonly_) {
      std::ofstream out(path, std::ios::binary);
      serialize_(obj, out);
    }
    return memcache_.Add(key, obj);
  }

  TypePtr Get(const K& key) override {
    TypePtr mem = memcache_.Get(key);
    if (mem) {
      return mem;
    }
    std::string path = GetPath(key);
    if (!Exists(path)) {
      TORCH_LAZY_COUNTER("PersistentCacheMiss", 1);
      return nullptr;
    }
    std::ifstream in(path, std::ios::binary);
    TypePtr val = deserialize_(in);
    if (!val) {
      TORCH_LAZY_COUNTER("PersistentCacheDeserializeFailure", 1);
      return nullptr;
    }
    TORCH_LAZY_COUNTER("PersistentCacheHit", 1);
    // Make sure the memcache tracks the value to prevent multiple loads
    return memcache_.Add(key, val);
  }

  void Clear() override {
    memcache_.Clear();
    // TODO(jonbolin): Clear the cache on disk
  }

  bool Erase(const K& key) override {
    memcache_.Erase(key);
    return !readonly_ && std::remove(GetPath(key).c_str());
  }

 private:
  std::string GetPath(K key) {
    std::stringstream ss;
    ss << cache_dir_ << "/" << key << ".bin";
    return ss.str();
  }

  bool Exists(std::string path) {
    struct stat buffer;
    return stat(path.c_str(), &buffer) == 0;
  }

  Cache<K, T, H, E> memcache_;
  std::function<void(TypePtr&, std::ostream&)> serialize_;
  std::function<TypePtr(std::istream&)> deserialize_;
  std::string cache_dir_;
  bool readonly_;
};

}  // namespace util
}  // namespace runtime
}  // namespace torch_xla

#endif  // XLA_CLIENT_CACHE_H_
