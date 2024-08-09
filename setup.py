from setuptools import setup, find_packages

setup(
    name="odvpkg",
    version="1.0",
    packages=find_packages(),
    author="Crey Indeer",
    author_email="spcrey@outlook.com",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/spcrey/odvpkg",
    install_requires=[
        "requests",
        "numpy",
        "matplotlib",
        "tqdm",
        "imageio",
        "netCDF4",
        "scipy"

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)