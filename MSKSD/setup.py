# setup.py
from setuptools import setup, find_packages

setup(
    name="msksd",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
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
