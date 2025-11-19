"""
MS-KSD-Bayes: Mode-Sensitive Kernel Stein Discrepancy for Bayesian Inference

A Python package implementing weighted Kernel Stein Discrepancy methods
for robust Bayesian inference with multimodal posteriors.
"""

__version__ = "0.1.0"
__author__ = "Elham Afzali"
__email__ = "afzalie@myumanitoba.ca"

# Core imports for easy access
from .src.KSD_Bayes import KSD_Bayes, KSDOutput, compute_weights
from .src.kernel import Scaled_PIMQ, KernelOutput
from .src.rscm_torch import RSCMEstimator

# Unimodal methods
from .Unimodal.Gauss import Gauss

# Bimodal methods  
from .Bimodal.KEF import KEF

__all__ = [
    'KSD_Bayes',
    'KSDOutput', 
    'compute_weights',
    'Scaled_PIMQ',
    'KernelOutput',
    'RSCMEstimator',
    'Gauss',
    'KEF'
]