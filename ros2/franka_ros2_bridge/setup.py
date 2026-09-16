import os
from glob import glob

from setuptools import Extension, find_packages, setup


package_name = "franka_ros2_bridge"


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
            sources=["src/cartesian_stream.cpp"],
            include_dirs=include_dirs,
            libraries=["franka"],
            language="c++",
            extra_compile_args=["-O3", "-std=c++17"],
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
