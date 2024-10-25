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

http_archive(
    name = "com_nlohmann_json",
    build_file = "//bazel:nlohmann_json.BUILD",
    sha256 = "d69f9deb6a75e2580465c6c4c5111b89c4dc2fa94e3a85fcd2ffcd9a143d9273",
    strip_prefix = "json-3.11.2",
    url = "https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz",
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")

# This is required for setting up the linkopts for -lpython.q
python_configure(
    name = "local_config_python",
    python_version = "3",  # required to use `python3-config`
)

################################ PyTorch Setup ################################

load("//bazel:dependencies.bzl", "PYTORCH_LOCAL_DIR")

new_local_repository(
    name = "torch",
    build_file = "//bazel:torch.BUILD",
    path = PYTORCH_LOCAL_DIR,
)

############################# OpenXLA Setup ###############################

# To update OpenXLA to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.

xla_hash = 'f8eded9d390de4c72b82f1e2bfaca9b8d737761c'

http_archive(
    name = "xla",
    patch_args = [
        "-l",
        "-p1",
    ],
    patch_tool = "patch",
    patches = [
        "//openxla_patches:gpu_race_condition.diff",
    ],
    strip_prefix = "xla-" + xla_hash,
    urls = [
        "https://github.com/openxla/xla/archive/" + xla_hash + ".tar.gz",
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

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    requirements = {
        "3.8": "//:requirements_lock_3_8.txt",
        "3.9": "//:requirements_lock_3_9.txt",
        "3.10": "//:requirements_lock_3_10.txt",
        "3.11": "//:requirements_lock_3_11.txt",
    },
    local_wheel_workspaces = ["@torch//:WORKSPACE"],
    default_python_version = "system",
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()



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

load("@tsl//third_party/gpus:cuda_configure.bzl", "cuda_configure")
cuda_configure(name = "local_config_cuda")
load("@tsl//third_party/nccl:nccl_configure.bzl", "nccl_configure")
nccl_configure(name = "local_config_nccl")
