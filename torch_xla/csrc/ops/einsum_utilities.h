#ifndef XLA_TORCH_XLA_CSRC_OPS_EINSUM_UTILITIES_H_
#define XLA_TORCH_XLA_CSRC_OPS_EINSUM_UTILITIES_H_

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "third_party/xla_client/debug_macros.h"

namespace torch_xla {

class EinsumUtilities {
 public:
  static std::vector<std::string> BuildBackwardsEquations(
      const std::string& equation) {
    std::vector<std::string> elements = ParseEquation(equation);
    std::vector<std::string> equations;
    equations.push_back(elements[2] + "," + elements[1] + "->" + elements[0]);
    equations.push_back(elements[0] + "," + elements[2] + "->" + elements[1]);
    return equations;
  }

  static std::string BuildBackwardsEquation(const std::string& equation) {
    int split_index = equation.find("->");
    XLA_CHECK_NE(split_index, std::string::npos);

    std::vector<std::string> elements = ParseEquation(equation);

    std::string backward_equation = elements[1] + "->" + elements[0];
    return backward_equation;
  }

  // An einsum equation is invalid if there are indices in one of the inputs or
  // output which are not in any other input or output. This is because such
  // equations lead to a failure when attempting to execute einsum backward on
  // XLA.
  static bool EquationIsValid(const std::string& equation) {
    // Elements represent the inputs and outputs of an equation
    // For example, in "i,j->ij" the elements are {"i", "j", "ij"}
    std::vector<std::string> elements = ParseEquation(equation);

    for (size_t i = 0; i < elements.size(); i++) {
      for (char c : elements[i]) {
        // We use j to skip searching the element for its own characters
        size_t j = 0;

        // For each element, we want to see if the chars in that element are
        // contained in some other element. For example, "i" is contained in
        // "ij", "j" is contained in "ij", and the characters in "ij" are
        // contained in "i" and "j" respectively, so the equation is valid. As a
        // counter-example, if we have "ik,ij->i", that is not valid, because
        // not all characters of "ik" or "ij" are contained in another element.
        if (std::all_of(elements.cbegin(), elements.cend(),
                        [&c, &i, &j](std::string element) {
                          bool elem_match = j++ == i;
                          bool char_match =
                              std::find(element.begin(), element.end(), c) ==
                              element.end();
                          return elem_match || char_match;
                        })) {
          return false;
        }
      }
    }
    return true;
  }

 private:
  // Breaks an einsum equation string down into its "elements", e.g. "i,j->ij"
  // will be decomposed into {"i", "j", "ij"}
  static std::vector<std::string> ParseEquation(const std::string& equation) {
    int split_index_one = equation.find(",");
    int split_index_two = equation.find("->");
    XLA_CHECK_NE(split_index_two, std::string::npos);

    std::vector<std::string> elements;

    if (split_index_one == std::string::npos) {
      elements.push_back(equation.substr(0, split_index_two));
      elements.push_back(equation.substr(
          split_index_two + 2, equation.size() - split_index_two - 2));
    } else {
      elements.push_back(equation.substr(0, split_index_one));
      elements.push_back(equation.substr(
          split_index_one + 1, split_index_two - split_index_one - 1));
      elements.push_back(equation.substr(
          split_index_two + 2, equation.size() - split_index_two - 2));
    }

    return elements;
  }
};
}  // namespace torch_xla

#endif  // XLA_TORCH_XLA_CSRC_OPS_EINSUM_UTILITIES_H_