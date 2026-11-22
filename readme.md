<div align="center">

# Comparative Metaheuristic Optimization of Federated Proximal for Regression in Heterogeneous Federated Learning Environments

**Michael Angello Qadosy Riyadi · Bernadus Anggo Seno Aji · Sulthonika Mahfudz Al Mujahidin · Nur Alifia Rustan · Adinda Mariasti Dewi · Kadek Gandhi Wahyu Jaya Suastika**

Department of Information Technology, Telkom University, Surabaya Campus, Indonesia

---

[![DOI](https://img.shields.io/badge/DOI-10.5935%2Fjetia.v12i61.4767-003366?style=flat-square)](https://doi.org/10.5935/jetia.v12i61.4767)
[![Journal](https://img.shields.io/badge/Journal-ITEGAM--JETIA_v.12_n.61_(2026)-003366?style=flat-square)](https://www.itegam-jetia.org)
[![License](https://img.shields.io/badge/License-CC_BY_4.0-555555?style=flat-square)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.11.9-555555?style=flat-square)](https://python.org)
[![Status](https://img.shields.io/badge/Status-Published_October_2026-2e7d32?style=flat-square)]()

</div>

---

## Authors

| Author | ORCID |
|:---|:---|
| Michael Angello Qadosy Riyadi | [0009-0003-0853-0989](https://orcid.org/0009-0003-0853-0989) |
| Bernadus Anggo Seno Aji | [0000-0002-8175-5629](https://orcid.org/0000-0002-8175-5629) |
| Sulthonika Mahfudz Al Mujahidin | [0009-0000-5873-1082](https://orcid.org/0009-0000-5873-1082) |
| Nur Alifia Rustan | [0009-0006-8714-5684](https://orcid.org/0009-0006-8714-5684) |
| Adinda Mariasti Dewi *(Corresponding)* | [0009-0008-1157-0478](https://orcid.org/0009-0008-1157-0478) |
| Kadek Gandhi Wahyu Jaya Suastika | [0009-0004-2370-215X](https://orcid.org/0009-0004-2370-215X) |

Correspondence: `dindamariasti@student.telkomuniversity.ac.id`

**Received:** April 10, 2026 | **Accepted:** June 15, 2026 | **Published:** October 31, 2026

---

## Table of Contents

1. [Abstract](#abstract)
2. [Background and Motivation](#background-and-motivation)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
   - [Federated Learning and FedProx](#federated-learning-and-fedprox)
   - [Optimization Methods](#optimization-methods)
   - [Experimental Setup](#experimental-setup)
5. [Results](#results)
6. [Conclusions](#conclusions)
7. [Repository Structure](#repository-structure)
8. [Getting Started](#getting-started)
9. [Citation](#citation)

---

## Abstract

Federated Learning (FL) enables collaborative model training without sharing raw data, but its performance often degrades under heterogeneous non-identically distributed (non-IID) client data. This study evaluates population-based metaheuristic algorithms to optimize the proximal hyperparameter (μ) in the Federated Proximal (FedProx) algorithm for regression tasks under highly heterogeneous conditions. Experiments used a structured tabular dataset of 1,338 observations from an individual health insurance dataset, chosen due to its sensitive medical and demographic attributes requiring strong privacy preservation. The data were partitioned into five heterogeneous clients, with non-IID severity validated using one-way ANOVA (F = 244.9053; p = 4.6609 × 10⁻¹⁴³). Five metaheuristic methods — Differential Evolution (DE), Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Harris Hawks Optimization (HHO), and Genetic Algorithm (GA) — were compared against Grid Search and Random Search using 10-fold cross-validation. Performance was evaluated using RMSE, MAE, R², search time, and memory usage. Results show that FedProx achieved stable convergence under extreme heterogeneity, with HHO obtaining the lowest RMSE (0.7653 ± 0.0561) and PSO the highest R² (0.1468 ± 0.1793) at round 50. These findings highlight the trade-off between optimization quality and computational efficiency.

**Keywords:** Federated Learning, Population-based Metaheuristic, Federated Proximal, Heterogeneous Data, Federated Regression

---

## Background and Motivation

The increasing adoption of machine learning across privacy-sensitive domains — healthcare, finance, and the Internet of Things — has intensified the need for distributed learning paradigms that preserve data locality. Federated Learning (FL) addresses this by enabling multiple clients to collaboratively train a shared model without transferring raw data to a central server, thus aligning with regulations such as the General Data Protection Regulation (GDPR). Despite its promise, FL faces a fundamental challenge: real-world client data are rarely identically distributed. This non-IID condition causes local model updates to diverge significantly, which can degrade global model accuracy by as much as 29% on benchmark datasets under strong heterogeneity.

FedProx extends the standard Federated Averaging (FedAvg) algorithm by introducing a proximal regularization term that constrains local parameter drift during each communication round. The effectiveness of this regularization, however, depends critically on the proximal coefficient μ. Prior literature has largely relied on manual tuning or simple heuristics for this parameter, without systematic adaptive frameworks — particularly in regression settings, which remain underexplored relative to classification. This study fills that gap by conducting a rigorous comparative evaluation of seven hyperparameter search strategies under extreme non-IID conditions, using a real-world health insurance dataset as the federated regression benchmark.

---

## Dataset

This study uses the publicly available individual health insurance claims dataset comprising 1,338 observations across seven predictor variables and one continuous target variable (`charges`, annual insurance cost in USD). The dataset contains no missing values. Its sensitive medical and demographic attributes make it a natural candidate for a privacy-preserving federated scenario.

**Predictor variables:** `age` (continuous), `bmi` (continuous), `children` (discrete), `sex` (categorical), `smoker` (categorical), `region` (categorical).

### Non-IID Client Partitioning

To simulate heterogeneous federated conditions, the dataset was partitioned into five clients based on `region × smoker` combinations, iteratively splitting the largest group until five clients were obtained. This yields the following distribution:

| Client | Region — Smoker | n | Mean Charges (USD) | Std. Dev. (USD) |
|:---:|:---|:---:|:---:|:---:|
| 1 | Northeast — Non-smoker | 257 | 9,166 | 6,157 |
| 2 | Northeast — Smoker | 67 | 29,674 | 11,590 |
| 3 | Northwest — Non-smoker | 267 | 8,556 | 6,129 |
| 4 | Northwest — Smoker | 58 | 30,192 | 11,414 |
| 5 | Southeast — Non-smoker | 273 | 8,032 | 6,137 |

The mean charges for smoker-dominated clients (2 and 4) are approximately 3.5–3.8 times higher than those for non-smoker clients, producing substantial inter-client distributional divergence. This extreme heterogeneity was confirmed by a one-way ANOVA test yielding F = 244.9053 and p = 4.6609 × 10⁻¹⁴³, validating that the client charge distributions differ significantly — a deliberately challenging non-IID scenario designed to stress-test FedProx's robustness.

---

## Methodology

### Federated Learning and FedProx

The standard FL objective minimizes a weighted average of local loss functions across K clients:

```
min_θ F(θ) = Σ p_k · F_k(θ),   where p_k = |D_k| / Σ|D_k'|
```

Under Federated Averaging (FedAvg), each client independently minimizes its local loss, which can lead to severe parameter drift when data are non-IID. FedProx addresses this by augmenting each client's local objective with a proximal penalty term:

```
F_k^prox(θ; θ^t) = F_k(θ) + (μ/2) · ‖θ − θ^t‖²
```

Here, θ^t is the global model at round t, and the proximal coefficient μ ≥ 0 controls the degree of regularization. When μ = 0, the formulation reduces to FedAvg. As μ increases, local updates are increasingly constrained to remain close to the global model, trading local specialization for global consistency. The optimal value of μ for a given heterogeneous scenario is non-trivial to determine — this is the central hyperparameter tuning problem addressed by this study.

### Optimization Methods

Seven hyperparameter search strategies were evaluated to find the optimal μ ∈ [0.001, 1.0]:

| Method | Category | Population Size | Max Iterations | Search Mechanism |
|:---|:---|:---:|:---:|:---|
| Differential Evolution (DE) | Metaheuristic | 50 | 100 | Difference-vector mutation and crossover |
| Particle Swarm Optimization (PSO) | Metaheuristic | 50 | 100 | Cognitive and social velocity updates |
| Grey Wolf Optimizer (GWO) | Metaheuristic | 50 | 100 | Leadership hierarchy encirclement |
| Harris Hawks Optimization (HHO) | Metaheuristic | 50 | 100 | Adaptive soft/hard encirclement with Lévy flights |
| Genetic Algorithm (GA) | Metaheuristic | 50 | 100 | Linear crossover and additive mutation |
| Grid Search | Deterministic baseline | — | 40 candidates | Exhaustive linear grid evaluation |
| Random Search | Stochastic baseline | — | 30 samples | Uniform random sampling |

The objective function for all methods is the mean generalization error across clients, defined as:

```
J(μ) = (1/K) Σ L_k(μ),   where L_k(μ) = 1 − R²_k(μ)
```

Minimizing J(μ) is equivalent to maximizing the average coefficient of determination across all five federated clients.

### Experimental Setup

**Local model architecture.** Each client trains a feedforward neural network for regression comprising three fully connected layers: a hidden layer of 64 neurons (ReLU), a second hidden layer of 32 neurons (ReLU), and a linear output layer producing continuous charge predictions.

**Validation framework.** To ensure statistically robust performance estimates, 10-fold cross-validation was applied independently at each client across 50 federated communication rounds. Performance was evaluated using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and the coefficient of determination (R²) for both local and aggregated global models.

**Hardware environment.** AMD Ryzen 5 6600H (12 cores, ~3.3 GHz), 16 GB RAM, Windows 11 Home 64-bit, Python 3.11.9, primary optimization library: `mealpy`.

---

## Results

### Convergence Behavior Across Communication Rounds

At communication round 1, global model performance is uniformly poor across all methods, with RMSE values near 1.0 and negative R² (ranging from −0.0649 to −0.0558). This indicates that the initial global model fails to outperform a simple mean predictor — a direct consequence of the severe inter-client heterogeneity. The 3.5–3.8× gap in mean charges between smoker and non-smoker clients causes substantial divergence in early local updates, undermining the initial aggregation.

By round 25, a marked improvement is observed: global RMSE drops to approximately 0.804–0.813 and R² becomes positive for most methods, reflecting the stabilizing influence of FedProx's proximal term. This term effectively limits local parameter drift, enabling more consistent weighted aggregation across heterogeneous clients. Convergence continues through round 50, where global RMSE stabilizes in the range 0.765–0.775 — a reduction of approximately 22–23% from round 1 — and the highest R² reaches 0.1468.

### Global Model Performance at Round 50

| Method | Global RMSE | Global MAE | Global R² | Search Time (s/fold) |
|:---|:---:|:---:|:---:|:---:|
| HHO | **0.7653 ± 0.0561** | 0.5981 ± 0.0376 | 0.1347 ± 0.2079 | 99.54 ± 8.87 |
| PSO | 0.7684 ± 0.0546 | 0.6089 ± 0.0451 | **0.1468 ± 0.1793** | 54.24 ± 4.37 |
| DE | 0.7686 ± 0.0599 | 0.5978 ± 0.0404 | 0.1257 ± 0.2170 | 54.36 ± 4.45 |
| GWO | 0.7721 ± 0.0707 | 0.6101 ± 0.0564 | 0.1167 ± 0.2501 | 54.47 ± 4.62 |
| GA | 0.7751 ± 0.0633 | 0.6099 ± 0.0501 | 0.1297 ± 0.1989 | 0.73 ± 0.06 |
| Grid Search | 0.7704 ± 0.0610 | 0.6012 ± 0.0489 | 0.1263 ± 0.2181 | 0.37 ± 0.04 |
| Random Search | 0.7750 ± 0.0708 | 0.6042 ± 0.0527 | 0.0981 ± 0.2640 | 0.33 ± 0.03 |

HHO achieves the lowest global RMSE (0.7653), attributable to its multi-phase adaptive encirclement strategy — combining soft and hard encirclement with Lévy-flight-based escapes and progressive rapid dives — which allows more effective navigation of the noisy, potentially multimodal objective landscape J(μ). PSO achieves the highest R² (0.1468), benefiting from its balanced integration of individual cognitive memory and social information sharing. DE performs closely behind PSO in RMSE, while GWO, GA, and both baselines exhibit marginally inferior convergence. Crucially, a Friedman test at round 50 yields non-significant results for all three metrics (RMSE: χ² = 6.17, p = 0.4043; MAE: χ² = 5.27, p = 0.5095; R²: χ² = 2.36, p = 0.8841), indicating that no method produces statistically superior predictive performance.

### Local vs. Global Model Performance

Local models consistently outperform their global counterparts, with median RMSE gaps of approximately 0.041–0.044, median MAE gaps of 0.06–0.07, and R² advantages of roughly 0.07–0.08. This persistent gap demonstrates that FedProx's weighted averaging has not fully reconciled the extreme inter-client heterogeneity, and that local specialization remains superior to the federated global consensus under these conditions. Outliers in global performance are most prevalent in folds where small client sizes (Clients 2 and 4, with 67 and 58 samples respectively) introduce high aggregation variance.

### Computational Efficiency

Computational costs vary dramatically across methods. HHO incurs the highest total search time (~50,137 seconds across 500 evaluations), approximately double that of DE, PSO, and GWO (~27,000–27,500 seconds each). GA, Grid Search, and Random Search complete in 165–371 seconds total. Memory usage is highest for DE (1,014.61 MB) and GWO (518.41 MB), while other methods consume less than 20 MB. The Friedman test confirms highly significant inter-method differences in both time (χ² = 56.14, p < 0.0001) and memory (χ² = 26.71, p = 0.0002). Post-hoc Nemenyi tests indicate HHO is significantly slower than nearly all competitors, while Grid Search and Random Search form the most efficient group. The elevated cost of HHO arises from its multi-phase coefficient computations and conditional branching per iteration, in contrast to the simpler update rules of PSO, DE, and GWO.

### Optimal μ Distribution

Adaptive methods (HHO and PSO) tend to select μ values that achieve a balanced trade-off between constraining local drift and permitting sufficient local adaptation — consistent with their marginally superior numerical performance. Exhaustive and stochastic baselines (Grid and Random Search) show wider μ distributions, reflecting their broader but less targeted exploration strategies.

---

## Conclusions

This study provides a systematic empirical evaluation of population-based metaheuristic algorithms for optimizing the proximal hyperparameter μ in FedProx under extreme non-IID federated regression conditions. The principal findings are as follows:

1. **FedProx convergence stability.** FedProx achieves stable and consistent convergence across all optimization strategies despite extreme inter-client distributional divergence (ANOVA F = 244.9053), reducing global RMSE by approximately 22–23% from round 1 to round 50. This confirms the effectiveness of proximal regularization in constraining local drift under severe non-IID conditions.

2. **Marginal but consistent metaheuristic advantages.** HHO and PSO yield the best numerical performance (lowest RMSE and highest R², respectively) among all evaluated methods, suggesting that adaptive exploration strategies confer some benefit in navigating the noisy, low-dimensional μ-tuning landscape. However, these gains are modest in absolute terms and statistically non-significant (Friedman test, all p > 0.05 at round 50).

3. **Computational trade-offs.** The performance advantages of HHO and PSO come at a substantial computational cost — up to two orders of magnitude greater than Grid Search or Random Search. In resource-constrained federated deployments, simpler strategies remain competitive while offering significant efficiency gains.

4. **Persistent local-global performance gap.** Local models consistently outperform the global model across all methods, with RMSE gaps of 0.041–0.044 at round 50. This gap reflects the inherent tension between client-level specialization and global model generalization in highly heterogeneous environments, and may require more advanced personalized federated learning strategies to resolve.

**Limitations and future directions.** This study is limited by its focus on a single hyperparameter in a one-dimensional search space, one structured dataset with synthetic partitioning, and a relatively simple local neural network architecture. Future work should investigate: (1) simultaneous multi-hyperparameter optimization using surrogate-based or dimensionality-reduction approaches; (2) more diverse real-world federated regression datasets from multi-institutional or cross-domain sources; (3) comparisons with state-of-the-art federated-specific tuning methods and advanced metaheuristic variants; and (4) broader evaluation dimensions including fairness, robustness, communication cost, and energy efficiency.

---

## Repository Structure

```
fedprox-metaheuristic-optimization-ITEGAMJETIA/
│
├── notebooks/
│   ├── 1__EDA_dataset_per_client.ipynb              # Exploratory data analysis per client
│   ├── 2__main_code.ipynb                           # Main experiment: FedProx + metaheuristics
│   ├── 3__results.ipynb                             # Result aggregation and statistical tests
│   └── 3b__lineplot_groundtruth_vs_prediksi.ipynb   # Ground truth vs. prediction visualization
│
├── data/
│   └── insurance.csv                                # Individual health insurance dataset
│
├── results/
│   └── fedprox_results.json                         # Raw experimental outputs (all methods)
│
├── paper/
│   └── ITEGAMJETIA_v12n61_2026_fedprox.pdf          # Published manuscript
│
├── README.md
└── requirements.txt
```

---

## Getting Started

### Prerequisites

```
Python >= 3.11.9
```

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/fedprox-metaheuristic-optimization-ITEGAMJETIA.git
cd fedprox-metaheuristic-optimization-ITEGAMJETIA
pip install -r requirements.txt
```

### Core Dependencies

```
numpy
pandas
scikit-learn
mealpy
torch
scipy
matplotlib
seaborn
jupyter
```

### Reproducing the Experiments

```bash
# Step 1 — Exploratory data analysis
jupyter notebook notebooks/1__EDA_dataset_per_client.ipynb

# Step 2 — Main federated experiment (FedProx + all 7 optimization methods)
jupyter notebook notebooks/2__main_code.ipynb

# Step 3 — Result aggregation, statistical tests, and figures
jupyter notebook notebooks/3__results.ipynb

# Step 4 — Ground truth vs. prediction line plots
jupyter notebook notebooks/3b__lineplot_groundtruth_vs_prediksi.ipynb
```

---

## Citation

If this code or manuscript contributes to your research, please cite both the published paper and this repository.

### Paper (ITEGAM-JETIA, 2026)

```bibtex
@article{riyadi2026fedprox,
  author    = {Riyadi, Michael Angello Qadosy and
               Aji, Bernadus Anggo Seno and
               {Al Mujahidin}, Sulthonika Mahfudz and
               Rustan, Nur Alifia and
               Dewi, Adinda Mariasti and
               Suastika, Kadek Gandhi Wahyu Jaya},
  title     = {Comparative Metaheuristic Optimization of Federated Proximal
               for Regression in Heterogeneous Federated Learning Environments},
  journal   = {Journal of Engineering and Technology for Industrial Applications
               (ITEGAM-JETIA)},
  volume    = {12},
  number    = {61},
  pages     = {1675--1688},
  year      = {2026},
  month     = oct,
  publisher = {Galileo Institute of Technology and Education of the Amazon (ITEGAM)},
  address   = {Manaus, Brazil},
  doi       = {10.5935/jetia.v12i61.4767},
  url       = {https://doi.org/10.5935/jetia.v12i61.4767},
  issn      = {2447-0228},
  note      = {Open Access, licensed under CC BY 4.0},
  keywords  = {federated learning, population-based metaheuristic,
               federated proximal, heterogeneous data, federated regression}
}
```

### Software Repository

```bibtex
@software{riyadi2026fedprox_code,
  author    = {Riyadi, Michael Angello Qadosy and
               Aji, Bernadus Anggo Seno and
               {Al Mujahidin}, Sulthonika Mahfudz and
               Rustan, Nur Alifia and
               Dewi, Adinda Mariasti and
               Suastika, Kadek Gandhi Wahyu Jaya},
  title     = {{fedprox-metaheuristic-optimization-ITEGAMJETIA}:
               Companion Source Code for Comparative Metaheuristic
               Optimization of FedProx for Regression in Heterogeneous
               Federated Learning Environments},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/YOUR_USERNAME/fedprox-metaheuristic-optimization-ITEGAMJETIA},
  version   = {1.0.0},
  license   = {CC-BY-4.0},
  note      = {Companion code for ITEGAM-JETIA, vol. 12, no. 61, pp. 1675--1688, 2026.
               DOI: 10.5935/jetia.v12i61.4767}
}
```

---

<div align="center">

[![DOI](https://img.shields.io/badge/DOI-10.5935%2Fjetia.v12i61.4767-003366?style=flat-square)](https://doi.org/10.5935/jetia.v12i61.4767)

Department of Information Technology, Telkom University, Surabaya Campus, Indonesia
Journal of Engineering and Technology for Industrial Applications (ITEGAM-JETIA), v.12 n.61, 2026

</div>