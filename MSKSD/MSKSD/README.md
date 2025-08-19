# **MS-KSD-Bayes**  
_A Repository for the Paper: "Correcting Mode Proportion Bias in Generalized Bayesian Inference via a Weighted Kernel Stein Discrepancy"_

## **Overview**
This repository is dedicated to the **MS-KSD-Bayes** method, as introduced in our paper:  
📄 **Title:** _Correcting Mode Proportion Bias in Generalized Bayesian Inference via a Weighted Kernel Stein Discrepancy_  
✍ **Authors:** _Elham Afzali, Saman Muthukumarana, and Liqun Wang_  
📚 **Journal:** Bayesian Analysis (Under Review)

**MS-KSD-Bayes** extends the standard **KSD-Bayes** method by incorporating a density-based weighting function to improve sensitivity to mode proportions in multimodal posteriors. It refines Generalized Bayesian Inference (GBI) for likelihood-free Bayesian inference, ensuring robust and accurate posterior estimates in both unimodal and multimodal settings.

## **Key Features**
- **Weighted Kernel Stein Discrepancy (MS-KSD):** Corrects mode proportion bias in KSD-based inference.
- **Density-Based Weighting Function:** Enhances sensitivity to multimodal posteriors.
- **Regularization for Computational Efficiency:** Reduces computational costs while maintaining accuracy.
- **Likelihood-Free Inference:** Suitable for complex Bayesian models with intractable likelihoods.

## **Code Availability**
🚧 **Code Coming Soon!** 🚧  
The implementation of **MS-KSD-Bayes** will be made publicly available after the peer-review process is completed.  
Stay tuned for updates!  

## **Planned Repository Structure**
Upon release, the repository will include:
```
├── msksd_bayes/          # Main module with core implementation
│   ├── KEF.py            # Kernel Exponential Family functions
│   ├── kernel.py         # Pairwise IMQ kernel functions
│   ├── KSD_Bayes.py      # KSD-Bayes implementation
│   ├── M_KEF.py          # Multiple KEF methods
│   ├── nearestSPD.py     # Nearest Symmetric Positive Definite matrix computation
│   ├── pdf_KEF.py        # Probability density estimation via KEF
│   ├── rscm_torch.py     # Regularized Sample Covariance Matrix (RSCM) estimator
│   ├── run_kef.py        # Script to execute KEF with KSD-Bayes
├── examples/             # Example scripts for using MS-KSD-Bayes
├── experiments/          # Benchmarking experiments from the paper
├── README.md             # This document
```

## **Citation**
If you find our work useful, please consider citing:
```
@article{Afzali2025,
  author = {Elham Afzali, Saman Muthukumarana, and Liqun Wang},
  title = {Correcting Mode Proportion Bias in Generalized Bayesian Inference via a Weighted Kernel Stein Discrepancy},
  journal = {Bayesian Analysis},
  year = {2025},
}
```

## **Contact & Updates**
For any questions, feel free to reach out or follow updates on:  
📧 Email: *afzalie@myumanitoba.ca*  
🔗 [Link to Paper (if available)]  
