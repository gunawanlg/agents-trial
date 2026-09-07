"""Legacy-compatible package metadata for Python 3.6 installers."""

from setuptools import find_packages, setup


setup(
    name="scorecard-segment-eval",
    version="0.1.0",
    description="Sub-population evaluation for logistic scorecards.",
    packages=find_packages(include=["scorecard_segment_eval", "scorecard_segment_eval.*"]),
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.17",
        "pandas>=1.1",
        "scikit-learn>=0.24",
        "scipy>=1.5",
        "dataclasses>=0.8; python_version<'3.7'",
    ],
    extras_require={"dev": ["pytest>=6.2"]},
)
