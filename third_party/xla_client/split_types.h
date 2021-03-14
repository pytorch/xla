#pragma once

#include <string>
#include <vector>

namespace xla {
namespace torch_xla {

/**
 * @brief Split function calls between two different callbacks depending
 *        upon the boolean return value of a predicate callback
 * @return Complete set of results in the original order of inputs
 *
 * TODO: Optional classification callback in order to call both true and false
 *       with classification info and reconstitute the results
 */
template <typename RESULT_T, typename CONTAINER_T, typename FUNC_T,
          typename CALL_T1, typename CALL_T2>
RESULT_T split_types(CONTAINER_T &all, FUNC_T predicate, CALL_T1 true_call,
                     CALL_T2 false_call) {
  std::vector<std::size_t> true_indexes;
  std::vector<std::size_t> false_indexes;
  std::vector<typename CONTAINER_T::value_type> true_items;
  std::vector<typename CONTAINER_T::value_type> false_items;

  true_indexes.reserve(all.size());
  false_indexes.reserve(all.size());
  true_items.reserve(all.size());
  false_items.reserve(all.size());
  std::size_t index = 0;
  for (auto &item : all) {
    if (predicate(item)) {
      true_indexes.emplace_back(index);
      true_items.emplace_back(std::move(item));
    } else {
      false_indexes.emplace_back(index);
      false_items.emplace_back(std::move(item));
    }
    ++index;
  }

  const std::size_t true_count = true_items.size();
  const std::size_t false_count = false_items.size();

  // TODO: 2-way multi-wait
  // Currently, however, operating on both devices is generally not
  // on the performance path
  RESULT_T true_results = true_count ? true_call(true_items) : RESULT_T();
  RESULT_T false_results = false_count ? false_call(false_items) : RESULT_T();

  // xxxx_items may have undergone a move and
  // now have undefined content
  assert(true_results.size() == true_count);
  assert(false_results.size() == false_count);

  RESULT_T results(all.size());

  for (std::size_t i = 0; i < true_indexes.size(); ++i) {
    results[true_indexes[i]] = std::move(true_results[i]);
  }
  for (std::size_t i = 0; i < false_indexes.size(); ++i) {
    results[false_indexes[i]] = std::move(false_results[i]);
  }
  return std::move(results);
}

}  // namespace torch_xla
}  // namespace xla
