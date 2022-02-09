from setuptools import setup
from importlib import util
from importlib.metadata import version

# TODO: Include zennit

# Import package descriptions (long) and basic installation requirements.
with open("README.md", "r") as f1, open("requirements.txt", "r") as f2:
    long_description = f1.read()
    REQUIRED = f2.read()
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
setup(
    name="quantus",
    version="0.0.11",
    description="A metrics toolkit to evaluate neural network explanations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    url="http://github.com/understandable-machine-intelligence-lab/Quantus",
    author="Anna Hedstrom",
    author_email="hedstroem.anna@gmail.com",
    keywords=["explainable ai", "xai", "machine learning", "deep learning"],
    license="CC BY-NC-SA 3.0",
    packages=["quantus"],
    zip_safe=False,
    python_requires=">=3.6",
)
