load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")

_RULE_DEPS = [
    (
        "pypi__pep517",
        "https://files.pythonhosted.org/packages/25/6e/ca4a5434eb0e502210f591b97537d322546e4833dcb4d470a48c375c5540/pep517-0.13.1-py3-none-any.whl",
        "31b206f67165b3536dd577c5c3f1518e8fbaf38cbc57efff8369a392feff1721",
    ),
    (
        "pypi__pip",
        "https://files.pythonhosted.org/packages/8a/6a/19e9fe04fca059ccf770861c7d5721ab4c2aebc539889e97c7977528a53b/pip-24.0-py3-none-any.whl",
        "ba0d021a166865d2265246961bec0152ff124de910c5cc39f1156ce3fa7c69dc",
    ),
    (
        "pypi__pip_tools",
        "https://files.pythonhosted.org/packages/0d/dc/38f4ce065e92c66f058ea7a368a9c5de4e702272b479c0992059f7693941/pip_tools-7.4.1-py3-none-any.whl",
        "4c690e5fbae2f21e87843e89c26191f0d9454f362d8acdbd695716493ec8b3a9",
    ),
    (
        "pypi__setuptools",
        "https://files.pythonhosted.org/packages/de/88/70c5767a0e43eb4451c2200f07d042a4bcd7639276003a9c54a68cfcc1f8/setuptools-70.0.0-py3-none-any.whl",
        "54faa7f2e8d2d11bcd2c07bed282eef1046b5c080d1c32add737d7b5817b1ad4",
    ),
    (
        "pypi__wheel",
        "https://files.pythonhosted.org/packages/7d/cd/d7460c9a869b16c3dd4e1e403cce337df165368c71d6af229a74699622ce/wheel-0.43.0-py3-none-any.whl",
        "55c570405f142630c6b9f72fe09d9b67cf1477fcf543ae5b8dcb1f5b7377da81",
    ),
    (
        "pypi__zipp",
        "https://files.pythonhosted.org/packages/da/55/a03fd7240714916507e1fcf7ae355bd9d9ed2e6db492595f1a67f61681be/zipp-3.18.2-py3-none-any.whl",
        "dce197b859eb796242b0622af1b8beb0a722d52aa2f57133ead08edd5bf5374e",
    ),
    # END: maintained by 'bazel run //tools/private/update_deps:update_pip_deps'
]

_GENERIC_WHEEL = """\
package(default_visibility = ["//visibility:public"])

load("@rules_python//python:py_library.bzl", "py_library")
load("@rules_python//python/private:glob_excludes.bzl", "glob_excludes")

py_library(
    name = "lib",
    srcs = glob(["**/*.py"]),
    data = glob(["**/*"], exclude=[
        # These entries include those put into user-installed dependencies by
        # data_exclude to avoid non-determinism.
        "**/*.py",
        "**/*.pyc",
        "**/*.pyc.*",  # During pyc creation, temp files named *.pyc.NNN are created
        "**/*.dist-info/RECORD",
        "BUILD",
        "WORKSPACE",
    ] + glob_excludes.version_dependent_exclusions()),
    # This makes this directory a top-level in the python import
    # search path for anything that depends on this.
    imports = ["."],
)
"""

# Collate all the repository names so they can be easily consumed
all_repo_names = [name for (name, _, _) in _RULE_DEPS]
record_files = {
    name: Label("@{}//:{}.dist-info/RECORD".format(
        name,
        url.rpartition("/")[-1].partition("-py3-none")[0],
    ))
    for (name, url, _) in _RULE_DEPS
}

def pypi_deps():
    """
    Fetch dependencies these rules depend on. Workspaces that use the pip_parse rule can call this.
    """
    for (name, url, sha256) in _RULE_DEPS:
        maybe(
            http_archive,
            name,
            url = url,
            sha256 = sha256,
            type = "zip",
            build_file_content = _GENERIC_WHEEL,
        )
