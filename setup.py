from setuptools import setup, find_packages

setup(
    name="health_multimodal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # Dependencies should ideally be moved from requirements.txt to here,
        # but for now we keep this minimal to facilitate local installation.
    ],
)
