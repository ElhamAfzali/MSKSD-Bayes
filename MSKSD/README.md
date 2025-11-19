# Correcting Mode Proportion Bias in Generalized Bayesian Inference via a Weighted Kernel Stein Discrepancy

**Authors:** Elham Afzali, Saman Muthukumarana, Liqun Wang

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A Python package implementing **Mode-Sensitive Kernel Stein Discrepancy for Bayesian Inference**, a novel method that corrects mode proportion bias in Generalized Bayesian Inference via weighted Kernel Stein Discrepancy.

## Overview

**MS-KSD-Bayes** extends the standard KSD-Bayes method by incorporating a density-based weighting function to improve sensitivity to mode proportions in multimodal posteriors. This approach ensures robust and accurate posterior estimates in both unimodal and multimodal settings for likelihood-free Bayesian inference.

### Key Features

- **🎯 Weighted Kernel Stein Discrepancy (MS-KSD)**: Corrects mode proportion bias in KSD-based inference
- **🔀 Multimodal Posterior Support**: Enhanced sensitivity to multimodal distributions
- **⚡ Computational Efficiency**: Regularized Sample Covariance Matrix (RSCM) estimation for improved numerical stability
- **🚀 GPU Acceleration**: Full PyTorch integration with CUDA support
- **📊 Comprehensive Experiments**: Ready-to-use experimental framework for research validation

## Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (optional, for acceleration)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/ElhamAfzali/MSKSD-Bayes.git
cd MSKSD-Bayes

# Install dependencies
pip install torch numpy pandas matplotlib scikit-learn tqdm

# Install the package in development mode
pip install -e .
```

### Dependencies

The package requires the following Python packages:

- `torch` (≥2.0.0)
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tqdm`

## Package Structure

```text
MSKSD/
├── src/                          # Core algorithms
│   ├── KSD_Bayes.py             # Main KSD-Bayes implementation
│   ├── kernel.py                # Pairwise IMQ kernel functions
│   └── rscm_torch.py            # Regularized Sample Covariance Matrix
├── Bimodal/                     # Bimodal posterior methods
│   ├── KEF.py                   # Kernel Exponential Family functions
│   ├── M_KEF.py                 # Multiple KEF methods
│   ├── nearestSPD.py            # SPD matrix projection
│   ├── pdf_KEF.py               # Density estimation via KEF
│   └── experiments/             # Bimodal experiments
│       ├── run_galaxy.py        # Galaxy dataset experiments
│       └── run_gene.py          # Gene expression experiments
├── Unimodal/                    # Unimodal posterior methods
│   ├── Gauss.py                 # Gaussian location model
│   ├── M_Gauss.py               # Gaussian M-functions
│   └── run_gauss.py             # Gaussian experiments
└── setup.py                    # Package installation
```

## Running Experiments

```python
# Run galaxy dataset experiment
cd Bimodal/experiments
python run_galaxy.py

# Run gene expression experiment
python run_gene.py

# Results will be saved in respective directories:
# - galaxy_results/
# - gene_results/
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: Elham Afzali
- **Email**: <afzalie@myumanitoba.ca>
- **Institution**: University of Manitoba
- **GitHub**: [@ElhamAfzali](https://github.com/ElhamAfzali)
