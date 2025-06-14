ARG ARCH=arm64
FROM debian:latest

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     autoconf automake build-essential curl wget \
     make openssh-client openssh-server pkg-config \
     python-is-python3 python3-dev python3-pip \
     software-properties-common \
     telnet time unzip vim zip zstd

RUN wget https://apt.llvm.org/llvm.sh \
  && bash llvm.sh 18 \
  && apt install -y clang-format-18 \
  && ln -sf /usr/bin/clang-18 /usr/bin/clang \
  && ln -sf /usr/bin/clang++-18 /usr/bin/clang \
  && ln -sf /usr/bin/lld-18 /usr/bin/lld \
  && ln -sf /usr/bin/clangd-18 /usr/bin/clangd \
  && ln -sf /usr/bin/clang-format-18 /usr/bin/clang-format

RUN wget https://github.com/bazelbuild/bazel/releases/download/7.4.1/bazel-7.4.1-linux-arm64 -O bazel \
  && chmod +x bazel \
  && mv bazel /usr/bin

CMD ["/bin/bash"]
