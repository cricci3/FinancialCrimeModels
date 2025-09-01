# Enhancing Financial Client Segmentation Models through Time-Series Clustering

This repository contains the official code for the Master thesis "Enhancing Financial Client Segmentation Models through Time-Series Clustering".

<p align="center">
  <img src="https://github.com/user-attachments/assets/80920cfa-6cbe-4faa-80b3-0b8c28f226f1" alt="timeseries" width="250" heigth="250"/>
  <img src="https://github.com/user-attachments/assets/5ba6e641-725f-4ba8-8326-ff2561c8a973" alt="leiden_02" width="250" heigth="250"/>
</p>

In this work, we propose a framework that can generate client segments in settings where only aggregate balance data are available, simulating scenarios in which detailed transaction records cannot be accessed. Utilizing simulated customer and transaction data, the study develops an unsupervised, graph-based framework for client segmentation. Account behavior is modeled through time series derived from financial data, from which conditional dependencies are inferred using the [*Sparse QUadratic approximation for Inverse Covariance* (SQUIC)](https://epubs.siam.org/doi/10.1137/17M1147615) and [SQUIC-Fit](https://ieeexplore.ieee.org/document/10091452) algorithms. These dependencies are represented as graphs and multiple clustering methods are applied to uncover communities of behaviorally similar clients. The evaluation shows that the proposed framework can generate client segments in settings where only aggregate balance data are available, simulating scenarios in which detailed transaction records cannot be accessed. The approach further demonstrates scalability across datasets of increasing size, indicating potential for application in realistic financial contexts.

## Motivation & contributions
Client segmentation is a fundamental task in financial services, enabling institutions to tailor products, enhance customer satisfaction and strengthen risk management. Traditional segmentation approaches often rely on static demographic or firmographic attributes, which fail to capture the behavioral diversity of clients. To address this limitation, this thesis advances customer segmentation methodologies by leveraging innovative techniques in time-series clustering.

## Data Sources
- [AMLSim](https://github.com/IBM/AMLSim) datasets with [100-1K-10K-100K](https://github.com/IBM/AMLSim/wiki/Download-Example-Data-Set) users can be downloaded from [here](https://www.dropbox.com/scl/fo/7g35w7wk7gglve627we3k/AHjP6pnCmV8M62L7RxTFtkU?rlkey=6ksx339ac9117onfx3l0g3fji&e=1&dl=0)

     - AMLSim100: 100 clients and 10,000 transactions;
     - AMLSim1K: 1,000 clients and 100,000 transactions;
     - AMLSim10K: 10,000 clients and 1,000,000 transactions;
     - AMLSim100K: 100\,000 clients and 10\,000\,000 transactions.

- Data generated with [PaySim tool](https://github.com/EdgarLopezPhD/PaySim) can be downloaded from [here](https://drive.google.com/drive/folders/1Alv9liWAcDfOHLTj3aKzQsdb8YFn1Fze?usp=drive_link)

   - PaySim100: 111 clients and 12,492 transactions;
  - PaySim1K: 1,026 clients and 103,884 transactions;
  - PaySim10K: 10,284 clients and 1,100,726 transactions;
  - PaySim100K: 102,249 clients and 10,900,690 transactions.

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
    ├── experiment1.py
    ├── experiment2_paysim.py
    ├── ...
    ├── Datasets
        ├── AMLSim
            ├── 100 users (with inside the csv files)
            ├── 1K users (with inside the csv files)
            ├── 10K users (with inside the csv files)
            ├── 100K users (with inside the csv files)
        ├── PaySim
            ├── 100 users (with inside the csv files)
            ├── 1K users (with inside the csv files)
            ├── 10K users (with inside the csv files)
            ├── 100K users (with inside the csv files)
    ```

    Each subfolder (e.g., `100`, `1K`, etc.) should contain the corresponding `.csv` files from the dataset.

4. **Run the Program**  
    You can run client segmentation with multiple clustering methods by executing:

    ```bash
    python experiment1.py
    ```

    To run client segmentation with Spectral Clustering on PaySim (fixed number of clusters), execute:
     ```bash
     python experiment2_paysim.py
     ```

    A corresponding Jupyter Notebook (```.ipynb```) is provided for each experiment, as graph visualization with the [Cosmograph](https://github.com/cosmograph-org/py_cosmograph) tool is supported only in notebooks and not in ```.py``` files.

5. **Input the Dataset Name**  
    When prompted, input the dataset name using the following format: `Name_Dimension`, for example `AMLSim_100` or `PaySim_100` (the input is not case sensitive).


6. **View Results**  
    For `experiment1.py` (AMLSim/PaySim – Multiple Clustering Methods):
   Output will look like:

   ```
   For lambda = 0.01:
       louvain:  PDensity = 0.3, Q = 0.3,  nCluster = 4, nIsolated = 0
       leiden:   PDensity = 0.33, Q = 0.27, nCluster = 5, nIsolated = 0
       dbscan:   PDensity = 0.23, Q = -0.0, nCluster = 1, nIsolated = 0
       spectral:   PDensity = 0.23, Q = -0.0, nCluster = 1, nIsolated = 0
   ```

   - `PDensity`: Average density between clusters
   - `Q`: Modularity score
   - `nCluster`: Number of clusters detected
   - `nIsolated`: Number of isolated nodes

    For `experiment2_paysim`:
   Output will look like:

   ```
   For lambda = 0.6 : 'nCluster': 2, 'ARI': -0.03, 'f1': 0.58
   For lambda = 0.5 : 'nCluster': 2, 'ARI': 1.0, 'f1': 1.0
   For lambda = 0.4 : 'nCluster': 2, 'ARI': 1.0, 'f1': 1.0
   ```

   - `ARI`: Adjusted Rand Index
   - `f1`: F1 Score
   - `nCluster`: Number of clusters (should be 2)

    When running the Jupyter Notebooks, an additional visualization of the graph is provided, with communities highlighted in different colors.    
