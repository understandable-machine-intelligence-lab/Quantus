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
with open("requirements.txt", "r") as f:
    REQUIRES = f.read()

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
}

# Define setup.
setup(
    name="quantus",
    version="0.1.1",
    description="A metrics toolkit to evaluate neural network explanations.",
    long_description=open("README.md", "r").read(),
    long_description_content_type="text/markdown",
    install_requires=REQUIRES,
    extras_require=EXTRAS,
    url="http://github.com/understandable-machine-intelligence-lab/Quantus",
    author="Anna Hedstrom",
    author_email="hedstroem.anna@gmail.com",
    keywords=["explainable ai", "xai", "machine learning", "deep learning"],
    license="CC BY-NC-SA 3.0",
    packages=find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    include_package_data=True,
)
