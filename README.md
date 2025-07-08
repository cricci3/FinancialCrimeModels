# FinancialCrimeModels

## How to Run

0. **Install SQUIC**
   
    Follow the [SQUIC User Manual](https://www.gitlab.ci.inf.usi.ch/SQUIC/gitlab-profile/-/blob/main/SQUIC_User_Manual.pdf) to install SQUIC Library.

1. **Install all required dependencies**
    
    Python 3.10 or higher is required, then install all required dependencies with:

    ```
    pip install -r requirements.txt
    ```

2. **Download the Dataset**  
   Use the links provided in [**Data Sources**](README.md#data-sources) section to download the datasets.

3. **Setup the Folder Structure**  
   In the root directory of the project, create a folder named `Datasets` with the following structure:

    ```
    FinancialCrimeModels
    ├── client_segmentation_1A.py
    ├── client_segmentation_1B.py
    ├── ...
    ├── Datasets
        ├── AMLSim
            ├── 100 users (with inside the csv files)
            ├── 1K users (with inside the csv files)
            ├── 10K users (with inside the csv files)
            ├── 100K users (with inside the csv files)
            ├── 1M users (with inside the csv files)
        ├── PaySim
            ├── 100 users (with inside the csv files)
            ├── 1K users (with inside the csv files)
            ├── 10K users (with inside the csv files)
            ├── 100K users (with inside the csv files)
            ├── 1M users (with inside the csv files)
    ```

    Each subfolder (e.g., `100`, `1K`, etc.) should contain the corresponding `.csv` files from the dataset.

4. **Run the Program**  
    - To run **spectral clustering on PaySim datasets** (fixed number of clusters `k=2`), execute:

        ```bash
        python client_segmentation_1A.py
        ```

   - To run **multi-method clustering on AMLSim datasets** (dynamic number of clusters), execute:

     ```bash
     python client_segmentation_1B.py
     ```

5. **Input the Dataset Name**  
    When prompted, input the dataset name using the following format: `Name_Dimension`, for example `AMLSim_100` or `PaySim_100` (the input is not case sensitive).


6. **View Results**  
    For `client_segmentation_1A.py` (PaySim – Spectral Clustering, `k=2`):
   Output will look like:

   ```
   For rho = 0.6 : {'spectral': {'nCluster': 2, 'ARI': -0.03, 'f1': 0.58}}
   For rho = 0.5 : {'spectral': {'nCluster': 2, 'ARI': 1.0, 'f1': 1.0}}
   For rho = 0.4 : {'spectral': {'nCluster': 2, 'ARI': 1.0, 'f1': 1.0}}
   ```

   - `ARI`: Adjusted Rand Index
   - `f1`: F1 Score
   - `nCluster`: Number of clusters (should be 2)
    
   For `client_segmentation_1B.py` (AMLSim – Multiple Clustering Methods):
   Output will look like:

   ```
   For rho = 0.01:
       louvain:  PDensity = 0.3,  Int Density = 0.75, Q = 0.3,  nCluster = 4, nIsolated = 0
       leiden:   PDensity = 0.33, Int Density = 0.69, Q = 0.27, nCluster = 5, nIsolated = 0
       dbscan:   PDensity = 0.23, Int Density = 0.24, Q = -0.0, nCluster = 1, nIsolated = 0
   ```

   - `PDensity`: Average density between clusters
   - `Int Density`: Average density within clusters
   - `Q`: Modularity score
   - `nCluster`: Number of clusters detected
   - `nIsolated`: Number of isolated nodes


## Data Sources
- [AMLSim](https://github.com/IBM/AMLSim) datasets with [100-1K-10K-100K-1M](https://github.com/IBM/AMLSim/wiki/Download-Example-Data-Set) users can be downloaded from [here](https://www.dropbox.com/scl/fo/7g35w7wk7gglve627we3k/AHjP6pnCmV8M62L7RxTFtkU?rlkey=6ksx339ac9117onfx3l0g3fji&e=1&dl=0)

    Anomaly %
    - 100 = 0.1% (18 out of 17144 transactions)
    - 1K = 0.15% (175 out of 117533 transactions)
    - 10K = 0.13% (1719 out of 1323234 transactions)
    - 100K = 0.14% (17052 out of 12476012 transactions)
    - 1M = 0.13% (162937 out of 124703184 transactions)

- Data generated with [PaySim tool](https://github.com/EdgarLopezPhD/PaySim) can be downloaded from [here](https://drive.google.com/drive/folders/1Alv9liWAcDfOHLTj3aKzQsdb8YFn1Fze?usp=drive_link)

   Anomaly %
    - 100 = 0.14% (18 out of 12492 transactions)
    - 1K = 0.12% (128 out of 103884 transactions)
    - 10K = 0.13% (1444 out of 1100726 transactions)
    - 100K = 0.13% (14376 out of 10900690 transactions)
    - 1M = 0.13% (143548 out of 111526310 transactions)

- [Libra Bank transaction graph](https://graphomaly.upb.ro/index.htm#datasets) over 3 months

## Thesis Overview

**Title:**

Enhancing Financial Crime Segmentation Models through Time-Series Clustering

**Abstract:**

Financial institutions are mandated to monitor client transactions as part of Anti-Money Laundering (AML) and financial fraud prevention efforts. Traditional transaction monitoring systems rely on predefined static rules to identify suspicious behavior. These systems typically incorporate customer segmentation models to adjust detection thresholds based on client risk profiles. However, conventional segmentation models often fail to capture complex behavioral patterns that are critical for effective fraud detection [1].

This thesis aims to advance customer segmentation methodologies by leveraging innovative techniques in feature selection and time-series clustering. Utilizing anonymized customer and transaction data from a financial institution, this work will develop unsupervised, graph-based [2] clustering approaches to generate more accurate and meaningful customer partitions [3]. The validity of these newly derived clusters will be systematically compared against the bank’s existing segmentation framework. Additionally, correlations between the proposed clustering models and various customer attributes, including risk classifications, will be analyzed to assess their potential impact on financial crime detection.

[1] J. Schmidt, D. Pasadakis, M. Sathe, and O. Schenk, “[GAMLNet: a graph based framework for the detection of money laundering](https://ssl.lu.usi.ch/entityws/Allegati/3010824_638529309691881843.pdf)”, 2024 11th IEEE Swiss Conference on Data Science (SDS), Zurich, Switzerland, 2024, pp. 241-245, doi: 10.1109/SDS60720.2024.00043.

[2] D. Pasadakis, M. Bollhöfer, and O. Schenk, “[Sparse quadratic approximation for graph learning](https://www.researchgate.net/publication/367727002_Sparse_Quadratic_Approximation_for_Graph_Learning)”, IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 11256-11269, 1 Sept. 2023, doi: 10.1109/TPAMI.2023.3263969

[3]  M. Lechekhab, D. Pasadakis, and O. Schenk, “[Multilevel diffusion based spectral graph clustering](https://www.researchgate.net/publication/390483229_Multilevel_Diffusion_Based_Spectral_Graph_Clustering)”, in IEEE High Performance Extreme Computing Conference, 23 - 27 September 2024.


