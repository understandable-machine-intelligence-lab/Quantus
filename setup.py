# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.

import importlib
from setuptools import setup, find_packages
from sys import version_info
from importlib import util

# Interpret the version of a package depending on if python>=3.8 vs python<3.8:
# Read: https://stackoverflow.com/questions/20180543/how-to-check-version-of-python-modules?rq=1.
if version_info[1] <= 7:
    import pkg_resources

    def version(s: str):
        return pkg_resources.get_distribution(s).version


else:
    from importlib.metadata import version

# Define library import by choice of ML framework (if neither torch or tensorflow is installed return an empty list).
if (
    util.find_spec("torch")
    and version("torch") >= "1.2"
    and util.find_spec("tensorflow")
    and version("tensorflow") >= "2.0"
):
    extras = ["captum==0.4.1", "tf-explain==0.3.1"]
elif util.find_spec("torch") and version("torch") >= "1.2":
    extras = ["captum==0.4.1"]
elif util.find_spec("tensorflow") and version("tensorflow") >= "2.0":
    extras = ["tf-explain==0.3.1"]
else:
    extras = []

# Define basic package imports.
# with open("requirements.txt", "r") as f:
#    REQUIRES = f.read()

# Define extras.
EXTRAS = {
    "torch": ["torch==1.10.1", "torchvision==0.11.2"],
    "tensorflow": ["tensorflow==2.6.2"],
    "extras": extras,
    "tutorials": [
        "torch==1.10.1",
        "torchvision==0.11.2",
        "captum==0.4.1",
        "collections",
        "pandas",
        "xmltodict",
        "xml",
    ],
    "zennit": [
        "torch==1.10.1",
        "torchvision==0.11.2",
        "zennit==0.4.5",
        "captum==0.4.1",
    ],
}

# Define setup.
setup(
    name="quantus",
    version="0.2.2",
    description="A metrics toolkit to evaluate neural network explanations.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib==3.3.4",
        "numpy>=1.19.5",
        "opencv-python==4.5.5.62",
        "protobuf~=3.19.0",
        "scikit-image==0.19.1",
        "scikit-learn==0.24.2",
        "scipy==1.7.3",
        "termcolor==1.1.0",
        "tqdm==4.62.3",
    ],
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
