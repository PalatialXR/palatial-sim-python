from setuptools import setup, find_packages

setup(
    name="segment_annotate",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pybullet",
        "sentence-transformers",
    ],
    python_requires=">=3.7",
) 