load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

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
        "//tf_patches:f16_abi_clang.diff",
        "//tf_patches:gpu_race_condition.diff",
        "//tf_patches:grpc_version.diff",
        "//tf_patches:stream_executor.diff",
        "//tf_patches:thread_local_random.diff",
        "//tf_patches:xplane.diff",
        "//tf_patches:topk_rewriter.diff",
        "//tf_patches:local_rendezvous.diff",
    ],
    strip_prefix = "tensorflow-5e229cfaa3c13c389f864b0ed38877fb61936161",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/5e229cfaa3c13c389f864b0ed38877fb61936161.tar.gz",
    ],
)

# local_repository(
#    name = "org_tensorflow",
#    path = "/workspace/pytorch/xla/third_party/tensorflow",
# )

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
