load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "utils",
    srcs = [
        "utils.py",
    ],
    deps = [
        "@jax-src//jax",
    ],
)

py_library(
    name = "pipelined_scan",
    srcs = [
        "pipelined_scan.py",
    ],
    deps = [
        ":utils",
        "//magics:ops",
        "@jax-src//jax",
        "@jax-src//jax/extend",
    ],
)
