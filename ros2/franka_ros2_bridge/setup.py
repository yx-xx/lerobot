import os
from glob import glob

from setuptools import Extension, find_packages, setup


package_name = "franka_ros2_bridge"

FRANKA_ROOT = os.environ.get("FRANKA_ROOT", "/home/rt/franka/libfranka")


def source_file(filename: str) -> str:
    """Find an extension source when colcon stages setup.py in build/.

    ament_python may execute a copied setup.py from build/<package>.  In that
    case the real source package remains at src/<package> in the workspace.
    """
    setup_dir = os.path.abspath(os.path.dirname(__file__))
    source_roots = (
        setup_dir,
        os.path.join(setup_dir, "..", "..", "src", package_name),
    )
    for source_root in source_roots:
        path = os.path.join(source_root, "src", filename)
        if os.path.isfile(path):
            return os.path.abspath(path)
    return os.path.join(setup_dir, "src", filename)


def cartesian_stream_extension() -> list[Extension]:
    if os.environ.get("FRANKA_SKIP_STREAM_EXT", "0") == "1":
        return []
    try:
        import pybind11
    except ImportError:
        # A system pybind11-dev package installs headers without a Python module.
        # Let the compiler find those headers and fail the build clearly if they
        # are unavailable. Silently omitting this extension creates a bridge that
        # installs successfully but cannot run in stream mode.
        include_dirs = []
    else:
        include_dirs = [pybind11.get_include()]
    return [
        Extension(
            "franka_ros2_bridge.cartesian_stream",
            sources=[source_file("cartesian_stream.cpp")],
            include_dirs=include_dirs + [os.path.join(FRANKA_ROOT, "include")],
            library_dirs=[os.path.join(FRANKA_ROOT, "lib")],
            libraries=["franka"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
            runtime_library_dirs=[os.path.join(FRANKA_ROOT, "lib")],
        )
    ]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}", ["README.md"]),
    ],
    ext_modules=cartesian_stream_extension(),
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Franka ROS 2 Bridge Maintainers",
    maintainer_email="maintainer@example.com",
    description="Reusable ROS 2 bridge for Franka robots using libfranka streaming.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge_node = franka_ros2_bridge.bridge_node:main",
        ],
    },
)
