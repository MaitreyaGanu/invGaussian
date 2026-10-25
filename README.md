# Estimating the Mean of a Selected Inverse Gaussian Population under Relative Squared Error Loss

![Topic](https://img.shields.io/badge/topic-selection%20bias-blue)
![Distribution](https://img.shields.io/badge/distribution-inverse%20Gaussian-green)
![Loss](https://img.shields.io/badge/loss-RSELF-orange)
![Status](https://img.shields.io/badge/code-coming%20soon-lightgrey)

Estimation after selection for inverse Gaussian (IG) populations. We select the population with the largest sample mean and estimate its **true** mean $`\mu_S`$ under relative squared error loss.

**Authors:** Maitreya Sameer Ganu, Dr. Md. Masihuddin, School of Data Science, IISER Thiruvananthapuram

---

## Table of Contents

- [Overview](#overview)
- [Problem Setup](#problem-setup)
- [Key Theoretical Results](#key-theoretical-results)
- [Estimators](#estimators)
- [Simulation Design](#simulation-design)
- [Results at a Glance](#results-at-a-glance)
- [Which Estimator Should I Use?](#which-estimator-should-i-use)
- [Figures](#figures)
- [Limitations and Extensions](#limitations-and-extensions)
- [Code Availability](#code-availability)

---

## Overview

When one population is picked because its sample mean is the largest, the selected sample mean tends to overestimate that population's true mean (the *winner's curse*). This project studies the problem for IG populations, where the variance $`\mu^3/\lambda`$ grows with the cube of the mean, so the populations most likely to be selected are also the noisiest.

**What the report does**

- Shows that the target $`\mu_S`$ is **random** and differs from the largest true mean $`\mu_{(k)}`$.
- Proves that the naive estimator has **strictly positive bias and relative bias** for every parameter configuration.
- Gives **exact one-dimensional integral expressions** for the bias, relative bias and risk of any estimator that depends on the data only through $`\bar X_S`$.
- Compares four estimators over a full grid of **216 configurations** $`(n, k, \rho)`$ by Monte Carlo.

---

## Problem Setup

| Item | Definition |
|---|---|
| Data | $`X_{ij} \overset{\text{iid}}{\sim} \mathrm{IG}(\mu_i, \lambda)`$, $`i=1,\dots,k`$, $`j=1,\dots,n`$ |
| Shape parameter | $`\lambda > 0`$, known and common to all populations |
| Sufficient statistic | $`\bar X_i \sim \mathrm{IG}(\mu_i, n\lambda)`$ |
| Selection rule | $`S = \arg\max_i \bar X_i`$ |
| Target | $`\mu_S = \sum_i \mu_i \, I(S=i)`$, the true mean of the selected population |
| Leading-means ratio | $`\rho = \mu_{(k)} / \mu_{(k-1)} \ge 1`$ |
| Loss (RSELF) | $`L(\delta, \mu_S) = \left(\delta/\mu_S - 1\right)^2`$ |
| Risk | $`R(\boldsymbol\mu,\delta) = \mathbb{E}\left[(\delta/\mu_S - 1)^2\right]`$ |
| Bias | $`B(\boldsymbol\mu,\delta) = \mathbb{E}(\delta - \mu_S)`$ |
| Relative bias | $`\beta(\boldsymbol\mu,\delta) = \mathbb{E}(\delta/\mu_S) - 1`$ |

---

## Key Theoretical Results

| Result | Statement |
|---|---|
| **Proposition 1** | $`\mathbb{E}(\mu_S) \le \mu_{(k)}`$, strictly unless all means are equal. The selected mean is not the largest true mean. |
| **Proposition 2** | For $`k \ge 2`$, the naive estimator has $`B > 0`$ and $`\beta > 0`$ for **all** $`\boldsymbol\mu`$, $`n`$, $`\lambda`$. |
| **Lemma 1** | For $`\delta = g(\bar X_S)`$, bias, relative bias and risk are sums of 1-D integrals of the form $`\sum_i \int_0^\infty h(x,\mu_i)\, f_{\mathrm{IG}}(x;\mu_i,n\lambda) \prod_{j\ne i} F_{\mathrm{IG}}(x;\mu_j,n\lambda)\,dx`$ |
| **Proposition 3** | The bias-corrected estimator always has smaller bias than the naive one, but is not guaranteed to be unbiased. |
| **Remark** | Removing the ordinary bias does **not** remove the relative bias, so it need not reduce the RSELF risk. |
| **Remark** | The generalized Bayes posterior is proper if and only if $`c > 2/3`$. |

---

## Estimators

| Estimator | Notation | Definition | Prior | Depends on |
|---|---|---|---|---|
| **Naive MLE** | $`\widehat\mu_{S,\mathrm{MLE}}`$ | $`\bar X_S`$ | None | $`\bar X_S`$ only |
| **Bias-corrected** | $`\widehat\mu_{S,\mathrm{BC}}`$ | $`\bar X_S - \widehat B_{\mathrm{MLE}}(\bar{\mathbf X})`$ (plug-in bias estimate) | None | all $`k`$ sample means |
| **Generalized Bayes** | $`\widehat\mu_{S,\mathrm{GB}}`$ | $`J_{3c/2-1}(\bar X_S) \big/ J_{3c/2}(\bar X_S)`$ | Extended Jeffreys, $`\pi \propto \prod_i \mu_i^{-3c/2}`$, $`c > 2/3`$ (improper) | $`\bar X_S`$ only |
| **Empirical Bayes** | $`\widehat\mu_{S,\mathrm{EB}}`$ | Bayes rule with $`(\widehat a,\widehat b)`$ estimated from all sample means | $`\mu_i \overset{\text{iid}}{\sim} \mathrm{Gamma}(a,b)`$ | all $`k`$ sample means |

---

## Simulation Design

| Quantity | Value |
|---|---|
| Sample size $`n`$ | 10, 50, 100, 500, 1000, 10000 |
| Number of populations $`k`$ | 5, 10, 50, 100, 250, 500 |
| Ratio $`\rho`$ | 1, 1.5, 2, 5, 7.5, 10 |
| Largest true mean $`\mu_{(k)}`$ | 5 |
| Second largest mean | $`\mu_{(k-1)} = \mu_{(k)}/\rho`$ |
| Remaining $`k-2`$ means | Equally spaced in $`[0.25\,\mu_{(k-1)},\ \mu_{(k-1)})`$ |
| Shape $`\lambda`$ | 5 (known) |
| Prior index $`c`$ (GB) | 3 |
| Monte Carlo replications $`M`$ | 10,000 per configuration |
| Total configurations | $`6\times6\times6 = 216`$ |
| Random seed | 1 |
| Integration | Gauss–Legendre quadrature |
| EB hyperparameters | L-BFGS-B on $`(\log a, \log b)`$, method-of-moments start |

> **Checks.** For the naive and generalized Bayes estimators, Monte Carlo results were checked against the exact 1-D integrals. The standard error of the estimated risk is at most about 0.002.

**Empirical Bayes fit diagnostics** (2,160,000 fits in total)

| Outcome | Count |
|---|---|
| Failed and excluded | 2 |
| Stopped before stationarity check was met | 11,156 (< 0.5%) |
| Ended on the box boundary | 10 |
| No method-of-moments start | 7 |

---

## Results at a Glance

### Head-to-head summary (216 configurations)

| | Naive MLE | Bias-corrected | Generalized Bayes | Empirical Bayes |
|---|:---:|:---:|:---:|:---:|
| **Configurations with smallest risk** | 0 | 59 | **116** | 41 |
| **Configurations with larger risk than naive** | n/a | 97 | **0** | 129 |
| **Worst-case risk over the grid** | 1.18 | 0.20 | **0.16** | 0.37 |
| **Typical bias direction** | Always positive | Reduces bias, can overcorrect | Negative in 83% of configurations | Negative in 82% of configurations |
| **Strongest when** | $`n`$ large or $`\rho \ge 5`$ | $`\rho = 1`$, $`n \ge 50`$ | Everywhere (safest) | $`k`$ small or $`\rho`$ near 1 |
| **Weakest when** | $`\rho`$ near 1 and $`n`$ small | $`n=10`$, large $`k`$; $`\rho=1.5`$, $`n=100`$, $`k=500`$ (+49% risk) | Slight underestimation | $`k \ge 50`$ and $`\rho \ge 5`$ (up to 9.3× naive risk) |

### Selection bias of the naive estimator (relative bias, $`k = 500`$)

| $`\rho`$ | $`n = 10`$ | $`n = 50`$ | $`n = 100`$ | $`n = 10000`$ |
|:---:|:---:|:---:|:---:|:---:|
| **1** | 105% | n/a | 25% | 1.5% |
| **2** | 29% | < 0.1% | < 0.1% | < 0.1% |
| **≥ 5** | bias < $`4\times10^{-5}`$ in every configuration | | | |

### Notable numbers

| Finding | Value |
|---|---|
| GB risk reduction vs. naive | up to **89%** |
| GB risk smaller than naive | **216 / 216** configurations |
| GB median absolute relative bias for $`\rho \ge 5`$ | about 2% |
| BC bias at $`\rho=1`$, $`k=500`$, $`n \ge 50`$ | 0.02 to 0.14 (naive: 0.07 to 1.80) |
| BC never worse than naive in risk at | $`\rho = 1`$ |
| EB risk at $`n=10`$, $`k=500`$, $`\rho=1`$ | 0.04 (naive: 1.18) |
| EB with smallest risk at $`k=5`$ | 25 of 36 configurations |
| Large samples, $`n = 10000`$ | all four estimators have risk < $`3\times10^{-4}`$ |

---

## Which Estimator Should I Use?

| Situation | Recommendation |
|---|---|
| Don't know $`\rho`$, want a safe default | **Generalized Bayes**. It never had larger risk than the naive estimator in the study. |
| Small bias is the main concern and the leading means are suspected to be close | **Bias-corrected** |
| Few populations ($`k`$ small) or close means | **Empirical Bayes** |
| Very large $`n`$ ($`\ge 10000`$) | Any of the four; the choice matters little |
| $`k`$ large and one mean clearly leads | Avoid **Empirical Bayes** |

---

## Figures

For each measure there are two sets of surfaces over $`(k, n)`$ (both on a log scale), with one panel per estimator: one for $`\rho \in \{1, 1.5, 2\}`$ and one for $`\rho \in \{5, 7.5, 10\}`$.

| Measure | Source file |
|---|---|
| Risk | `v_Surface_Risk_by_k_and_n.pdf` |
| Bias | `v_Surface_Bias_by_k_and_n.pdf` |
| Absolute bias | `v_Surface_AbsBias_by_k_and_n.pdf` |
| Relative bias | `v_Surface_RelBias_by_k_and_n.pdf` |

<!-- GitHub does not render PDFs inline. Convert the pages to PNG and uncomment, for example:
![Risk, rho=1](figures/risk_rho1.png)
-->

---

## Limitations and Extensions

**Limitations**

- Results come from one choice of $`\lambda=5`$, $`c=3`$, $`\mu_{(k)}=5`$ and one spacing of the remaining means. They should not be extrapolated beyond this grid without further study.
- Assumes a known common shape parameter, equal sample sizes, and selection of the largest sample mean.
- The plug-in bias and risk estimates are evaluated at the fitted means, so they are not unbiased for the bias and risk at the true $`\boldsymbol\mu`$.
- The bias-corrected estimator is not guaranteed to be positive.

**Possible extensions**

- Unknown shape parameter $`\lambda`$
- Unequal sample sizes
- Selecting the smallest mean, or the best $`t`$ populations
- Other loss functions
- Admissibility and minimaxity of the proposed rules

---

## Code Availability

Code is available in codes/IG.py
