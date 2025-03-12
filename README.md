# FinancialCrimeModels

## Thesis Overview

**Title:**

Enhancing Financial Crime Segmentation Models through Time-Series Clustering

**Abstract:**

Financial institutions are mandated to monitor client transactions as part of Anti-Money Laundering (AML) and financial fraud prevention efforts. Traditional transaction monitoring systems rely on predefined static rules to identify suspicious behavior. These systems typically incorporate customer segmentation models to adjust detection thresholds based on client risk profiles. However, conventional segmentation models often fail to capture complex behavioral patterns that are critical for effective fraud detection [1].

This thesis aims to advance customer segmentation methodologies by leveraging innovative techniques in feature selection and time-series clustering. Utilizing anonymized customer and transaction data from a financial institution, this work will develop unsupervised, graph-based [2] clustering approaches to generate more accurate and meaningful customer partitions [3]. The validity of these newly derived clusters will be systematically compared against the bank’s existing segmentation framework. Additionally, correlations between the proposed clustering models and various customer attributes, including risk classifications, will be analyzed to assess their potential impact on financial crime detection.

[1] J. Schmidt, D. Pasadakis, M. Sathe, and O. Schenk, “GAMLNet: a graph based framework for the detection of money laundering,” 2024 11th IEEE Swiss Conference on Data Science (SDS), Zurich, Switzerland, 2024, pp. 241-245, doi: 10.1109/SDS60720.2024.00043.

[2] D. Pasadakis, M. Bollhöfer, and O. Schenk, “Sparse quadratic approximation for graph learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 11256-11269, 1 Sept. 2023, doi: 10.1109/TPAMI.2023.3263969

[3]  M. Lechekhab, D. Pasadakis, and O. Schenk, “Multilevel diffusion based spectral graph clustering,” in IEEE High Performance Extreme Computing Conference, 23 - 27 September 2024.

## Sources
- AMLSim 100-1K-10K-100K-1M users www.github.com/IBM/AMLSim/wiki/Download-Example-Data-Set

- AMLSim 10k users on kaggle www.kaggle.com/datasets/anshankul/ibm-amlsim-example-dataset

## Results
with 100 users

### SQUIC Results
| Parameter  | 0.1 | 0.01 | 0.001 | 0.0001 | 1e-05 |
|-------------|-----|-------|--------|---------|--------|
| **NNZ**      | 10000 | 10000 | 10000 | 10000 | 10000 |
| **NNZ/Row**  | 100   | 100   | 100   | 100    | 100   |
| **Time (s)** | 0.98  | 0.92  | 0.96  | 0.95   | 0.94  |

### SQUIC FIT Results
| Parameter  | 0.1 | 0.01 | 0.001 | 0.0001 | 1e-05 |
|-------------|-----|-------|--------|---------|--------|
| **NNZ**      | 4992 | 4992 | 4992 | 4992 | 4992 |
| **NNZ/Row**  | 49.92 | 49.92 | 49.92 | 49.92 | 49.92 |
| **Time (s)** | 1.91  | 1.87  | 1.85  | 1.89   | 1.85  |

### SQUIC FIT NORM Results
| Parameter  | 0.1 | 0.01 | 0.001 | 0.0001 | 1e-05 |
|-------------------|-----|-------|--------|---------|--------|
| **NNZ**             | 1482 | 2306 | 4170 | 4826 | 4872 |
| **NNZ/Row**         | 14.82 | 23.06 | 41.7 | 48.26 | 48.72 |
| **Time (s)**        | 1.89  | 18.26 | 47.39 | 47.46 | 47.44 |

