from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

#with open("requirements.txt", "r") as f:
#    required_packages = f.read().splitlines()

setup(
    name="quantus",
    version="0.0.1",
    description="A metrics toolkit to evaluate neural network explanations.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=['torch',
                      'torchvision',
                      'numpy',
                      'matplotlib',
                      'scikit-learn',
                      'scikit-image',
                      'scipy',
                      'opencv-python',
                      'captum',
                      'pytest',
                      'termcolor',
                      'pytest-lazy-fixture',
                      'coverage'],
    url="http://github.com/understandable-machine-intelligence-lab/Quantus",
    author="Anna Hedström; Franz Motzkus",
    author_email="hedstroem.anna@gmail.com",
    keywords=["explainable ai", "xai", "machine learning", "deep learning"],
    license="MIT",
    packages=["quantus"],
    zip_safe=False,
    python_requires=">=3.6",
)
