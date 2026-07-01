# Federated Learning for Intrusion Detection in Industrial IoT

**Performance analysis of FedAvg on the CIC IIoT 2025 dataset under realistic non-IID conditions, communication constraints, and partial client participation.**

> Undergraduate Thesis (CS4490Z) — Department of Computer Science, Western University
> Author: Candice Williams | Supervisor: Prof. Zubair Fadlullah | March 2026

[![NumPy](https://img.shields.io/badge/NumPy-4DABCF?logo=numpy&logoColor=fff)](#)
[![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?logo=matplotlib&logoColor=fff)](#)
[![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=fff)](#)
[![Scikit-learn](https://img.shields.io/badge/-scikit--learn-%23F7931E?logo=scikit-learn&logoColor=white)](#)
[![Seaborn](https://img.shields.io/badge/Seaborn-4EAEAA?logo=python&logoColor=fff)](#)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?logo=python)](https://www.python.org/)

---

## Description

This project evaluates whether Federated Learning (FL) is a practical alternative to centralized machine learning for intrusion detection in Industrial Internet of Things (IIoT) networks. Using the newly released CIC IIoT 2025 dataset, a lightweight feedforward neural network was trained with the FedAvg algorithm across 48 experimental configurations to measure how data heterogeneity, communication rounds, local training epochs, and client participation rates affect detection accuracy. The study is among the first to evaluate FL on this dataset using non-IID distributions modeled on realistic industrial attack profiles rather than artificial data skew.

## Motivation

Centralized IDS require aggregating raw traffic and sensor data from every industrial site, which raises bandwidth, latency, and data-governance concerns, especially when that data contains proprietary process information. Federated Learning allows multiple industrial facilities to collaboratively train a shared detection model without ever exposing their raw data. This project asks a practical question: **how much detection performance do you actually give up by keeping data local, and under what conditions is that tradeoff worth it?**

## Research Question

How do non-IID data distributions, communication parameters, and partial client participation affect federated intrusion detection performance in heterogeneous IIoT environments?

## Methodology

- **Dataset:** CIC IIoT 2025 (Firouzi et al., 2025) — 30,030 samples across 8 classes (Benign, Brute Force, DDoS, DoS, Malware, MITM, Reconnaissance, Web exploitation), reduced to 17 features per the dataset's published feature-selection guidance.
- **Model:** Feedforward MLP, ~11K parameters (Input(17) → 128 → 64 → 8), kept intentionally small so per-round communication cost (~44 KB per client update) reflects realistic IIoT edge-device constraints.
- **Federated setup:** FedAvg (McMahan et al., 2017) implemented as a standalone PyTorch simulation, with both IID (stratified) and non-IID (label-skew 0.7 / 0.9) client partitioning to simulate industrial sites with specialized attack profiles.
- **Experiment design:** 7 experiment groups varying client count (K), local epochs (E), communication rounds (R), data skew, and participation rate, each run with 3 random seeds (42, 123, 456) — 48 configurations total.
- **Baselines:** Centralized training (upper bound) and local-only isolated training (lower bound) establish the performance ceiling and floor that FL is measured against.

## Key Results

| Method | Accuracy | Macro F1 | Gap Closed |
|---|---|---|---|
| Centralized (Adam, 100 epochs) | 0.920 ± 0.006 | 0.885 ± 0.010 | 100.0% |
| FedAvg, IID, K=5, E=10, R=50 | 0.869 ± 0.004 | 0.798 ± 0.012 | 62.3% |
| FedAvg, Non-IID (skew=0.7) | 0.848 ± 0.010 | 0.752 ± 0.018 | 42.3% |
| FedAvg, Non-IID (skew=0.9) | 0.825 ± 0.004 | 0.721 ± 0.007 | 28.8% |
| Local-only (isolated training) | — | 0.655 ± 0.006 | 0.0% |

- FedAvg under IID conditions recovers **48–62%** of the centralized-vs-local performance gap.
- **Moderate non-IID skew (0.7) only costs 1.8% F1** — FedAvg is more resilient to realistic data heterogeneity than expected. Severe skew (0.9) costs more (4.5% F1).
- **Partial client participation at just 30% retains 98.6%** of full-participation performance, meaning industrial sites can join opportunistically without hurting the global model.
- **Local epochs (E), not communication rounds, is the highest-impact hyperparameter** — investing in local compute matters more than network bandwidth for FL-IDS deployments.

## Repository Structure

```
├── data/               # Preprocessed data and partitions for the CIC IIoT 2025 dataset
├── figures/            # List of figures explaining findings used in report
├── notebooks/          # Jypter notebooks where all analysis was conducted
├── results/            # Configs for the 7 experiment groups (EXP1–EXP7), Logged metrics, figures, and result tables
├── src/                # Python helper scripts
└── report/             # Full thesis PDF
```

## Tech Stack

Python, PyTorch, NumPy, pandas, scikit-learn (preprocessing/scaling), Jyputer

## Limitations & Future Work

This study focuses solely on FedAvg; algorithms like FedProx could further narrow the gap under severe non-IID skew. Differential privacy was out of scope, leaving the privacy-utility tradeoff unquantified. See the full report for a detailed discussion of threats to validity and proposed extensions (personalized FL, adaptive aggregation, DP-SGD).

## Full Report

The complete thesis, including related work, detailed methodology, per-class performance breakdowns, and discussion, is available in [`report/FinalReport_CandiceWilliams.pdf`](./report/FinalReport_CandiceWilliams.pdf).

## Citation

If you reference this work, please cite:

> Williams, C. (2026). *Federated Learning for Intrusion Detection in Industrial Internet of Things: Performance Analysis on the CIC IIoT 2025 Dataset.* Undergraduate Thesis, Department of Computer Science, Western University.
