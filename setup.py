from setuptools import setup, find_packages

setup(
    name="shipsim",
    version="1.3.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "scipy",
    ],
)
