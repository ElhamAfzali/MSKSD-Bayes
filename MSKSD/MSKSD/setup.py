# setup.py
from setuptools import setup, find_packages

setup(
    name="msksd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
    ],
    author="Elham Afzali",
    author_email="Elhaam.afzali@gmail.com",
    description="Correcting Mode Proportion Bias in Generalized Bayesian Inference via a Weighted Kernel Stein Discrepancy",
    url="https://github.com/ElhamAfzali/MSKSD-Bayes",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
