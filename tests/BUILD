load("@rules_python//python:defs.bzl", "py_test")
load("@pypi_local//:requirements.bzl", "requirement")

py_test(
    name = "ops_test",
    srcs = ["ops_test.py"],
    deps = [
        "//magics:ops",
        "//magics:cuda",
        "@jax-src//jax",
        "@jax-src//jaxlib/cuda:cuda_gpu_support",
        "@jax-src//jaxlib/cuda:cuda_plugin_extension",
        "@jax-src//jax_plugins/cuda:cuda_plugin",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)

py_test(
    name = "pipelined_scan_test",
    srcs = ["pipelined_scan_test.py"],
    deps = [
        "//magics:cuda",
        "//magics/transforms:pipelined_scan",
        "@jax-src//jax",
        "@jax-src//jax/extend",
        "@jax-src//jaxlib/cuda:cuda_gpu_support",
        "@jax-src//jaxlib/cuda:cuda_plugin_extension",
        "@jax-src//jax_plugins/cuda:cuda_plugin",
        requirement("absl-py"),
        requirement("numpy"),
    ],
)
