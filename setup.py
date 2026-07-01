from setuptools import setup, find_packages

setup(
    name="smd-mamba",
    version="0.1.0",
    description="SMD-Mamba for scanner-robust Alzheimer's diagnosis from 3D MRI",
    author="Arshpreet Kaur",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
