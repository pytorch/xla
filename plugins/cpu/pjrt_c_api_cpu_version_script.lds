# Only symbols in the global section are available to other frameworks.
VERS_1.0 {
  global:
    extern "C" {
      GetPjrtApi;
    };

  local:
    *;
};
