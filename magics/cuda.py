import importlib
import os 
import ctypes

def load_nvidia_lib(module, lib_name):
    try:
        # Equivalent to `from nvidia import module`
        m = importlib.import_module(f"nvidia.{module}")
    except ImportError:
        raise ImportError(f"NVIDIA {module} are not installed.")
    path = os.path.join(m.__path__[0], f"lib/{lib_name}")
    # Makes sure this is called before other things start to
    # populate dlopen cache.
    ctypes.CDLL(path)


def initialize_cuda():
    # TODO: Load all the necessary NVIDIA libraries automatically.
    load_nvidia_lib("cudnn", "libcudnn.so.9")
    load_nvidia_lib("cufft", "libcufft.so.11")
    load_nvidia_lib("cublas", "libcublas.so.12")
    load_nvidia_lib("cusolver", "libcusolver.so.11")
    load_nvidia_lib("cusparse", "libcusparse.so.12")
    load_nvidia_lib("nccl", "libnccl.so.2")
    load_nvidia_lib("nvjitlink", "libnvJitLink.so.12")
    load_nvidia_lib("nvshmem", "libnvshmem_host.so.3")
    load_nvidia_lib("cuda_cupti", "libcupti.so.12")
