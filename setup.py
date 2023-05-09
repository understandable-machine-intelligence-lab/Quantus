# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.

import os
from setuptools import setup, find_packages
from importlib import util

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:

    required = f.read().splitlines()

with open(os.path.join(os.path.dirname(__file__), "requirements_test.txt")) as f:
    required_tests = f.read().splitlines()

# Define extras.
EXTRAS = {}
EXTRAS["torch"] = (
    [
        "torch<=1.11.0; python_version == '3.7'",
        "torch>=1.13.1; sys_platform != 'linux' and python_version > '3.7'",
        "torch>=1.13.1, <2.0.0; sys_platform == 'linux' and python_version > '3.7' and python_version <= '3.10'",
        "torch>=2.0.0; sys_platform == 'linux' and python_version >= '3.11'",
        "torchvision<=0.12.0; python_version == '3.7'",
        "torchvision>=0.15.1; sys_platform != 'linux' and python_version > '3.7'",
        "torchvision>=0.14.0, <0.15.1; sys_platform == 'linux' and python_version > '3.7' and python_version <= '3.10'",
        "torchvision>=0.15.1; sys_platform == 'linux' and python_version >= '3.11'",
     ]
    if not (util.find_spec("torch"))
    else []
)
EXTRAS["tensorflow"] = (
    [
        "tensorflow>=2.5.0; python_version == '3.7'",
        "tensorflow>=2.12.0; sys_platform != 'darwin' and python_version > '3.7'",
        "tensorflow_macos>=2.12.0; sys_platform == 'darwin' and python_version > '3.7'",
    ]
    if not (util.find_spec("tensorflow"))
    else []
)
EXTRAS["captum"] = (
    (EXTRAS["torch"] + ["captum>=0.6.0"]) if not util.find_spec("captum") else []
)
EXTRAS["tf-explain"] = (
    (EXTRAS["tensorflow"] + ["tf-explain>=0.3.1"])
    if not util.find_spec("tf-explain")
    else []
)
EXTRAS["zennit"] = (
    (EXTRAS["torch"] + ["zennit>=0.5.1"]) if not util.find_spec("zennit") else []
)
EXTRAS["tests"] = required + required_tests[1:]
EXTRAS["full"] = EXTRAS["captum"] + EXTRAS["tf-explain"] + EXTRAS["zennit"]

# Define setup.
setup(
    name="quantus",
    version="0.4.0",
    description="A toolkit to evaluate neural network explanations.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=required,
    extras_require=EXTRAS,
    url="http://github.com/understandable-machine-intelligence-lab/Quantus",
    author="Anna Hedstrom",
    author_email="hedstroem.anna@gmail.com",
    keywords=["explainable ai", "xai", "machine learning", "deep learning"],
    license="GNU LESSER GENERAL PUBLIC LICENSE VERSION 3",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    include_package_data=True,
)
