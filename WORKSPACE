load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

################################ Python Setup ################################

# For embedded python interpreter (libpython.so.)
http_archive(
    name = "pybind11_bazel",
    strip_prefix = "pybind11_bazel-fc56ce8a8b51e3dd941139d329b63ccfea1d304b",
    urls = ["https://github.com/pybind/pybind11_bazel/archive/fc56ce8a8b51e3dd941139d329b63ccfea1d304b.zip"],
)

http_archive(
    name = "pybind11",
    build_file = "@pybind11_bazel//:pybind11.BUILD",
    strip_prefix = "pybind11-442261da585536521ff459b1457b2904895f23b4",
    urls = ["https://github.com/pybind/pybind11/archive/442261da585536521ff459b1457b2904895f23b4.tar.gz"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

# This is required for setting up the linkopts for -lpython.q
python_configure(
    name = "local_config_python",
    python_version = "3",  # required to use `python3-config`
)
############################# OpenXLA Setup ###############################

# To update OpenXLA to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "xla",
    patch_args = [
        "-l",
        "-p1",
    ],
    patch_tool = "patch",
    patches = [
        "//openxla_patches:cache_urls.diff",
        "//openxla_patches:constexpr_return.diff",
        "//openxla_patches:gpu_race_condition.diff",
        "//openxla_patches:f16_abi_clang.diff",
        "//openxla_patches:gpu_topk_rewriter.diff",
    ],
    strip_prefix = "xla-4f8381651977dff16b1d86bb4b198eb733c5f478",
    urls = [
        "https://github.com/openxla/xla/archive/4f8381651977dff16b1d86bb4b198eb733c5f478.tar.gz",
    ],
)

# For development, one often wants to make changes to the OpenXLA repository as well
# as the PyTorch/XLA repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the OpenXLA repository on the build.py command line by passing a flag
#    like:
#    bazel --override_repository=xla=/path/to/openxla
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "xla",
#    path = "/path/to/openxla",
# )

# Initialize OpenXLA's external dependencies.
load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

################################ PyTorch Setup ################################

load("//bazel:dependencies.bzl", "PYTORCH_LOCAL_DIR")

new_local_repository(
    name = "torch",
    build_file = "//bazel:torch.BUILD",
    path = PYTORCH_LOCAL_DIR,
)
