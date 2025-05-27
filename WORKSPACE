workspace(name="jax_examples")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
    name = "xla",
    commit = "ada5b6e3187f173c5a7e7869fdf689287c3dfd6d",
    remote = "https://github.com/yliu120/xla.git",
)

git_repository(
    name = "jax-src",
    commit = "5567b58ec4c33ddc5976231938c9cb5ddbe6ab03",
    remote = "https://github.com/jax-ml/jax.git",
)

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    requirements = {
        "3.10": "@jax-src//build:requirements_lock_3_10.txt",
        "3.11": "@jax-src//build:requirements_lock_3_11.txt",
        "3.12": "@jax-src//build:requirements_lock_3_12.txt",
        "3.13": "@jax-src//build:requirements_lock_3_13.txt",
        "3.13-ft": "@jax-src//build:requirements_lock_3_13_ft.txt",
    },
    local_wheel_inclusion_list = [
        "jaxlib*",
        "jax_cuda*",
        "jax-cuda*",
    ],
    local_wheel_workspaces = ["@jax-src//jaxlib:jax.bzl"],
    local_wheel_dist_folder = "../dist",
    default_python_version = "system",
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "pypi_local",
    requirements_lock = "//third_party/pypi_local:requirements.txt",
)
load("@pypi_local//:requirements.bzl", "install_deps")
install_deps()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

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

load("@jax-src//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

load("@jax-src//jaxlib:jax_python_wheel.bzl", "jax_python_wheel_repository")

jax_python_wheel_repository(
    name = "jax_wheel",
    version_key = "_version",
    version_source = "@jax-src//jax:version.py",
)

load(
    "@xla//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)

python_wheel_version_suffix_repository(
    name = "jax_wheel_version_suffix",
)

load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@xla//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@xla//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@xla//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load("//third_party/pip_nvidia:deps.bzl", "pypi_deps")
pypi_deps()

load("//third_party/pip_nvidia:pip.bzl", "nvidia_library")

nvidia_library(
    name = "nvidia-libs",
    wheels = [
        "nvidia-cublas-cu12==12.8.4.1",
        "nvidia-cuda-cupti-cu12==12.8.90",
        "nvidia-cuda-nvcc-cu12==12.8.93",
        "nvidia-cuda-runtime-cu12==12.8.90",
        "nvidia-cudnn-cu12==9.8.0.87",
        "nvidia-cufft-cu12==11.3.3.83",
        "nvidia-cusolver-cu12==11.7.3.90",
        "nvidia-cusparse-cu12==12.5.8.93",
        "nvidia-nccl-cu12==2.26.2",
        "nvidia-nvjitlink-cu12==12.8.93",
        "nvidia-nvshmem-cu12==3.2.5",
    ],
)
