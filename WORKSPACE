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

python_configure(name = "local_config_python")
############################# TensorFlow Setup ###############################

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "org_tensorflow",
    patch_args = [
        "-l",
        "-p1",
    ],
    patch_tool = "patch",
    patches = [
        "//tf_patches:cache_urls.diff",
        "//tf_patches:cudnn_int8x32.diff",
        "//tf_patches:f16_abi_clang.diff",
        "//tf_patches:gpu_race_condition.diff",
        "//tf_patches:grpc_version.diff",
        "//tf_patches:stream_executor.diff",
        "//tf_patches:thread_local_random.diff",
        "//tf_patches:xplane.diff",
    ],
    strip_prefix = "tensorflow-f7759359f8420d3ca7b9fd19493f2a01bd47b4ef",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/f7759359f8420d3ca7b9fd19493f2a01bd47b4ef.tar.gz",
    ],
)

# For development, one often wants to make changes to the TF repository as well
# as the PyTorch/XLA repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the TF repository on the build.py command line by passing a flag
#    like:
#    bazel --override_repository=org_tensorflow=/path/to/tensorflow
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "org_tensorflow",
#    path = "/path/to/tensorflow",
# )

# Initialize TensorFlow's external dependencies.
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1()

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

################################ PyTorch Setup ################################

load("//bazel:dependencies.bzl", "PYTORCH_LOCAL_DIR")

new_local_repository(
    name = "torch",
    build_file = "//bazel:torch.BUILD",
    path = PYTORCH_LOCAL_DIR,
)
