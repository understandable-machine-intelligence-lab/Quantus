# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.

from setuptools import setup, find_packages
from importlib import util
from importlib.metadata import version

# Define extras.
EXTRAS = {}
EXTRAS["torch"] = (
    ["torch==1.13.1", "torchvision==0.15.1"]
    if not (util.find_spec("torch") and version("torch") >= "1.2")
    else []
)
EXTRAS["tensorflow"] = (["tensorflow>=2.12.0"]
    if not (util.find_spec("tensorflow") and version("tensorflow") >= "2.0")
    else []
)
EXTRAS["captum"] = (
    (EXTRAS["torch"] + ["captum==0.6.0"]) if not util.find_spec("captum") else []
)
EXTRAS["tf-explain"] = (
    (EXTRAS["tensorflow"] + ["tf-explain==0.3.1"])
    if not util.find_spec("tf-explain")
    else []
)
EXTRAS["zennit"] = (
    (EXTRAS["torch"] + ["zennit==0.5.0"]) if not util.find_spec("zennit") else []
)
EXTRAS["tutorials"] = (
    EXTRAS["torch"] + EXTRAS["captum"] + ["pandas", "xmltodict", "tensorflow-datasets"]
)
EXTRAS["tests"] = EXTRAS["captum"] + EXTRAS["tf-explain"] + EXTRAS["zennit"]
EXTRAS["full"] = EXTRAS["tutorials"] + EXTRAS["tf-explain"] + EXTRAS["zennit"]

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Define setup.
setup(
    name="quantus",
    version="0.3.4",
    description="A metrics toolkit to evaluate neural network explanations.",
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
    python_requires=">=3.8",
    include_package_data=True,
)
