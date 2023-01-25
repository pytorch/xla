load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To update TensorFlow to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/tensorflow/tensorflow/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "org_tensorflow",
    patch_args = ["-p1"],
    patch_tool = "patch",
    patches = [
        "//tf_patches:f16_abi_clang.diff",
        "//tf_patches:stream_executor.diff",
        "//tf_patches:thread_local_random.diff",
        "//tf_patches:xplane.diff",
        "//tf_patches:bazel.diff",
    ],
    sha256 = "a056b236c032a86ea0f361d6a7f97319e43e33144d8aaf44e42f0d154fde5c0a",
    strip_prefix = "tensorflow-29f29e5874ec25b5f26c990c3981dcd56288fd0c",
    urls = [
        "https://github.com/tensorflow/tensorflow/archive/29f29e5874ec25b5f26c990c3981dcd56288fd0c.tar.gz",
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
