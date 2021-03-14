#pragma once

#include <algorithm>
#include <sstream>
#include <string>
#include <strstream>

namespace torch_xla {

/**
 * @brief Utility class to map an environment variable into
 *        a file-specific variable, such as a 'verbose' flag
 *        to enable/disable log level in a particular souce
 *        file at runtime.
 *
 *        The final incarnartion  is the macro
 *        VERBOSE_FILE
 *        at the bottom, along with a usage descrption.
 *
 */
class EnvFileMacro {
  static bool is_true(const std::string &s) {
    if (s.empty()) {
      return false;
    }
    const int c = ::toupper(s[0]);
    return c == 'Y' || c == 'T' || std::atoi(s.c_str()) > 0;
  }

  static bool get_env_bool(const std::string &name, bool default_value) {
    const char *s = getenv(name.c_str());
    if (!s || !*s)
      return default_value;
    return is_true(s);
  }

  template <class T>
  static T base_name(T const &path, T const &delims = "/\\") {
    return path.substr(path.find_last_of(delims) + 1);
  }
  template <class T> static T remove_extension(T const &filename) {
    typename T::size_type const p(filename.find_last_of('.'));
    return p > 0 && p != T::npos ? filename.substr(0, p) : filename;
  }
  /**
   * @brief Make macro-like name for environment variables
   *
   * @param file
   * @return std::string
   */
  static std::string file_to_macro_name(const std::string &file_name,
                                        const std::string &prefix) {
    std::stringstream ss;
    if (!prefix.empty()) {
      ss << prefix << "_";
    }
    std::string result = remove_extension(base_name(file_name));
    std::transform(result.begin(), result.end(), result.begin(), ::toupper);
    ss << result;
    return ss.str();
  }

public:
  /**
   * @brief Get a boolean from the environment variable based on a file name
   *        i.e. /usr/local/lib/my_file.hh -> VERBOSE_MY_FILE environment
   * variable
   *
   * @param file_name
   * @param default_value
   * @return true
   * @return false
   */
  static bool get_file_env_bool(const std::string &file_name,
                                bool default_value = false,
                                const std::string &prefix = "") {
    return get_env_bool(file_to_macro_name(file_name, prefix), default_value);
  }
};

/**
 * @brief Return a boolean value based upon whether the source file should
 * produce verbose output. Usage example: bool verbose = VERBOSE_FILE(false);
 *
 *        Then within the file's code, check the 'verbose' variable as needed.
 *        To set a file as verbose, set the environment variable formed from
 *        the file name:
 *
 *        i.e. /usr/local/lib/my_file.hh -> VERBOSE_MY_FILE
 *
 *        So, in this case:  'export VERBOSE_MY_FILE=1' causes verbose output
 */
#define VERBOSE_FILE(__dflt)                                                   \
  EnvFileMacro::get_file_env_bool(__FILE__, __dflt, "VERBOSE")

} // namespace torch_xla
