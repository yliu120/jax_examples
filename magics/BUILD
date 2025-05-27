load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "ops",
    srcs = ["ops.py"],
    deps = [
        "@jax-src//jax",
        "@jax-src//jax/extend",
    ]
)

py_library(
    name = "cuda",
    srcs = ["cuda.py"],
    deps = [
        "@nvidia-libs//:libs",
    ]
)