from glob import glob
from setuptools import find_packages, setup


package_name = "franka_ros2_bridge"

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
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Franka ROS 2 Bridge Maintainers",
    maintainer_email="maintainer@example.com",
    description="Reusable ROS 2 bridge for Franka robots using frankx.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "bridge_node = franka_ros2_bridge.bridge_node:main",
        ],
    },
)
