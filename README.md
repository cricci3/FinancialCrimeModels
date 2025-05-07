# FinancialCrimeModels

## How to Run

0. **Install SQUIC**
   
    Follow the [SQUIC User Manual](https://www.gitlab.ci.inf.usi.ch/SQUIC/gitlab-profile/-/blob/main/SQUIC_User_Manual.pdf) to install SQUIC Library.

2. **Install all required dependencies**
    
    Python 3.10 or higher is required, then install all required dependencies with:

    ```
    pip install -r requirements.txt
    ```

3. **Download the Dataset**  
   Use the links provided in [**Data Sources**](README.md#data-sources) section to download the datasets.

4. **Setup the Folder Structure**  
   In the root directory of the project, create a folder named `Datasets` with the following structure:

    ```
    FinancialCrimeModels
    |_ BenchClientSegment.py
    |_ BenchClientSegment_labels.py
    |_ ...
    |_ Datasets
        |_ AMLSim
            |_ 100 (with inside the csv files)
            |_ 1K (with inside the csv files)
            |_ 10K (with inside the csv files)
            |_ 100K (with inside the csv files)
            |_ 1M (with inside the csv files)
        |_ PaySim
            |_ 100 (with inside the csv files)
            |_ 1K (with inside the csv files)
            |_ 10K (with inside the csv files)
            |_ 100K (with inside the csv files)
            |_ 1M (with inside the csv files)
    ```

    Each subfolder (e.g., `100`, `1K`, etc.) should contain the corresponding `.csv` files from the dataset.

5. **Run the Program**  
    Execute either `BenchClientSegment.py` (`notebook.ipynb` if you want to visualize the graph with Cosmograph) or `BenchClientSegment_labels.py` depending on your needs.

6. **Input the Dataset Name**  
    When prompted, input the dataset name using the following format: `Name_Dimension`, for example `AMLSim_100` or `PaySim_100` (the input is not case sensitive).


7. **View Results**  
    Once the program runs, it will output results in the following format:
    ```
    For lambda = 0.001 : {'ncut': 3.28, 'rcut': 292.87, 'modularity': 0.16, 'CC': 2}
    For lambda = 0.01 : {'ncut': 2.85, 'rcut': 99.36, 'modularity': 0.25, 'CC': 2}
    For lambda = 0.02 : {'ncut': 2.73, 'rcut': 60.61, 'modularity': 0.27, 'CC': 2}
    ...
    ```

Where:
- `ncut`: Normalized Cut
- `rcut`: Ratio Cut
- `modularity`: Community modularity score
- `CC`: Number of connected components

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

[1] J. Schmidt, D. Pasadakis, M. Sathe, and O. Schenk, “GAMLNet: a graph based framework for the detection of money laundering,” 2024 11th IEEE Swiss Conference on Data Science (SDS), Zurich, Switzerland, 2024, pp. 241-245, doi: 10.1109/SDS60720.2024.00043.

[2] D. Pasadakis, M. Bollhöfer, and O. Schenk, “Sparse quadratic approximation for graph learning,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 9, pp. 11256-11269, 1 Sept. 2023, doi: 10.1109/TPAMI.2023.3263969

[3]  M. Lechekhab, D. Pasadakis, and O. Schenk, “Multilevel diffusion based spectral graph clustering,” in IEEE High Performance Extreme Computing Conference, 23 - 27 September 2024.


