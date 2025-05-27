"""nvidia_library implementation for WORKSPACE setups."""
load(":deps.bzl", "all_repo_names")

def _construct_pypath(mrctx, *, entries):
    """Helper function to construct a PYTHONPATH.

    Contains entries for code in this repo as well as packages downloaded from //python/pip_install:repositories.bzl.
    This allows us to run python code inside repository rule implementations.

    Args:
        mrctx: Handle to the module_ctx or repository_ctx.
        entries: The list of entries to add to PYTHONPATH.

    Returns: String of the PYTHONPATH.
    """

    if not entries:
        return None

    os = "linux"
    separator = ";" if "windows" in os else ":"
    pypath = separator.join([
        str(mrctx.path(entry).dirname)
        # Use a dict as a way to remove duplicates and then sort it.
        for entry in sorted({x: None for x in entries})
    ])
    return pypath

def _create_repository_execution_environment(rctx, python_interpreter):
    """Create a environment dictionary for processes we spawn with rctx.execute.

    Args:
        rctx (repository_ctx): The repository context.
        python_interpreter (path): The resolved python interpreter.
        logger: Optional logger to use for operations.
    Returns:
        Dictionary of environment variable suitable to pass to rctx.execute.
    """

    env = {
        "PYTHONPATH": _construct_pypath(
            rctx,
            entries = rctx.attr._python_path_entries,
        ),
    }
    return env

def _nvidia_library_impl(rctx):
    # In the repository rule, we have to rely on system python to bootstrap.
    if rctx.attr.name == "nvidia":
        fail("Change the name of the nvidia repository to something else.")

    python = rctx.which("python")
    args = [
        python,
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--force-reinstall",
        "-t",
        ".",
    ] + rctx.attr.wheels

    result = rctx.execute(
        # The command is the interpreter followed by our script and its args.
        args,
        quiet = False,
        # Manually construct the PYTHONPATH since we cannot use the toolchain here.
        environment = _create_repository_execution_environment(rctx, python),
    )

    if result.return_code != 0:
        fail(
            "Pip install failed. (Code: {}): {}".format(
                result.return_code,
                result.stderr,
            ),
        )
    
    rctx.file("BUILD", """
# A generic build file for NVIDIA precompiled libraries.

filegroup(
    name = "precompiled",
    srcs = glob([
        "**/*.so*",
        "**/*.h",
        "**/bin/*",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "libs",
    visibility = ["//visibility:public"],
    srcs = glob(["nvidia/**/*.py"]),
    data = [":precompiled"],
    imports = ["{name}"],
)

""".format(name=rctx.attr.name))

nvidia_library = repository_rule(
    attrs = {
        "wheels": attr.string_list(
            mandatory = True,
            doc = "List of nvidia wheel requirements, for example, nvidia-cublas-cu12==x.y.z",
        ),
        "_python_path_entries": attr.label_list(
            default = [
                Label("//:BUILD.bazel"),
            ] + [
                Label("@" + repo + "//:BUILD.bazel")
                for repo in all_repo_names
            ],
        ),
    },
    implementation = _nvidia_library_impl,
    doc = """Create a package containing nvidia libraries.""",
)
