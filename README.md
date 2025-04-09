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

## Data Sources
- AMLSim datasets with [100-1K-10K-100K-1M](www.github.com/IBM/AMLSim/wiki/Download-Example-Data-Set) users

    Anomaly %
    - 100 = 0.1% (18 out of 17144 transactions)
    - 1K = 0.15% (175 out of 117533 transactions)
    - 10K = 0.13% (1719 out of 1323234 transactions)
    - 100K = 0.14% (17052 out of 12476012 transactions)
    - 1M = 0.13% (162937 out of 124703184 transactions)

- Data generated from [PaySim](https://github.com/EdgarLopezPhD/PaySim) tool
  The generated datasets can be downloaded from [here](https://drive.google.com/drive/folders/1Ebt9SNsPrbM4rMkJGrZHw66EmvjVtIph?usp=sharing, https://drive.google.com/drive/folders/1JRdPMLmCf8zXLP5shaKSwXWU7p7pHS7C?usp=sharing, https://drive.google.com/drive/folders/1T2ug-ZyBd-b1gBgvAy91GGIy6gt4aX9y?usp=sharing, https://drive.google.com/drive/folders/1uLpdq0sP96s6CfyHPLdWHQcOk2tO-ZfU?usp=sharing, https://drive.google.com/drive/folders/1yRSB-d2KnTcx5ElEmUYQ2_QwHj5_2Z1Q?usp=sharing)

- [Libra Bank transaction graph](https://graphomaly.upb.ro/index.htm#datasets) over 3 months


