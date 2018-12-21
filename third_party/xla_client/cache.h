#ifndef TENSORFLOW_COMPILER_XLA_RPC_CACHE_H_
#define TENSORFLOW_COMPILER_XLA_RPC_CACHE_H_

#include <functional>
#include <list>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace xla {
namespace util {

// Generic key and object cache with LRU expiration policy.
template <typename K, typename T, typename H = std::hash<K>,
          typename E = std::equal_to<K>>
class Cache {
  using Element = std::pair<K, T>;
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

 public:
  explicit Cache(size_t max_size) : max_size_(max_size) {}

  // Adds an object to the cache, unless it already exists. If the cache grows
  // beyond the limit set during construction, the oldest used object will be
  // removed from the cache.
  void Add(K key, T object) {
    std::lock_guard<std::mutex> slock(lock_);
    element_list_.emplace_front(Element(std::move(key), std::move(object)));
    auto it = element_list_.begin();
    if (!element_map_.emplace(&it->first, it).second) {
      element_list_.erase(it);
    } else if (element_list_.size() > max_size_) {
      Element* last = &element_list_.back();
      element_map_.erase(&last->first);
      element_list_.pop_back();
    }
  }

  // Retrieves the existing object if it exists. If it does, it's position in
  // the LRU list gets moved to the head of the list.
  // Returns nullptr if no object with the specified key is found within the
  // cache.
  const T* Get(const K& key) {
    std::lock_guard<std::mutex> slock(lock_);
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      return nullptr;
    }
    if (it->second != element_list_.begin()) {
      // LRU re-positioning.
      element_list_.splice(element_list_.begin(), element_list_, it->second);
    }
    return &it->second->second;
  }

  bool Erase(const K& key) {
    std::lock_guard<std::mutex> slock(lock_);
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      return false;
    }
    auto lit = it->second->second;
    element_map_.erase(it);
    element_list_.erase(lit);
    return true;
  }

  void Clear() {
    std::lock_guard<std::mutex> slock(lock_);
    element_map_.clear();
    element_list_.clear();
  }

 private:
  std::mutex lock_;
  size_t max_size_ = 0;
  ElementList element_list_;
  ElementMap element_map_;
};

}  // namespace util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_RPC_CACHE_H_
