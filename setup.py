# This file is part of Quantus.
# Quantus is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
# Quantus is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
# You should have received a copy of the GNU Lesser General Public License along with Quantus. If not, see <https://www.gnu.org/licenses/>.
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
    from importlib.metadata import version  # noqa

# Define extras.
EXTRAS = {}  # noqa

EXTRAS["torch"] = (
    ["torch>=1.13.1", "torchvision==0.14.1"] if not util.find_spec("torch") else []
)
EXTRAS["tensorflow"] = (
    ["tensorflow>=2.11.0"] if not util.find_spec("tensorflow") else []
)
EXTRAS["captum"] = (
    EXTRAS["torch"] + ["captum==0.6.0"] if not util.find_spec("captum") else []
)
EXTRAS["tf-explain"] = (
    EXTRAS["tensorflow"] + ["tf-explain==0.3.1"]
    if not util.find_spec("tf-explain")
    else []
)
EXTRAS["zennit"] = (
    EXTRAS["torch"] + ["zennit>=0.5.0"] if not util.find_spec("zennit") else []
)
EXTRAS["tutorials"] = (
    EXTRAS["torch"]
    + EXTRAS["captum"]
    + ["pandas", "xmltodict", "tensorflow-datasets", "jupyter"]
)
EXTRAS["tests"] = EXTRAS["captum"] + EXTRAS["tf-explain"] + EXTRAS["zennit"]
EXTRAS["nlp"] = ["nlpaug>=1.1.11", "nltk>=3.8.1", "transformers>=4.26.0"]
EXTRAS["full"] = (
    EXTRAS["tutorials"] + EXTRAS["tf-explain"] + EXTRAS["zennit"] + EXTRAS["nlp"]
)

# Define setup.
setup(
    name="quantus",
    version="0.3.4",
    description="A metrics toolkit to evaluate neural network explanations.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib>=3.6.3",
        "numpy>=1.23.5",
        "opencv-python>=4.7.0.68",
        "scikit-image>=0.19.1",
        "scikit-learn>=1.2.1",
        "scipy>=1.10.0",
        "tqdm>=4.64.1",
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
