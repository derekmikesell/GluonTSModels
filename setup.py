from setuptools import setup, find_packages

setup(
    name="gluontsmodels",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "gluonts>=0.13.2",
        "torch>=1.13"
        # Add your dependencies here, for example:
        # "gluonts>=0.6.0",
        # "numpy>=1.18.0",
        # "pandas>=1.0.0",
    ],
    author="Derek Mikesell",
    author_email="derekmikesell@gmail.com",
    description="A package for GluonTS models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/GluonTSModels",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
