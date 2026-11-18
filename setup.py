"""Setup script for QMI-CFS."""
from pathlib import Path

from setuptools import find_packages, setup

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="qmi-cfs",
    version="0.1.0",
    description=(
        "Quantum-inspired mutual information for causal feature selection "
        "and treatment effect estimation"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Janith Ramanayake",
    url="https://github.com/JanithRamanayake523/Quantum-Mutual-Information-for-Causal-Feature-Selection",
    license="MIT",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "statsmodels>=0.14.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        # Without BorutaPy, BorutaFeatureSelector silently falls back to MI.
        "boruta": ["BorutaPy>=0.3.0"],
        "dev": ["pytest>=7.0", "pytest-cov"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
