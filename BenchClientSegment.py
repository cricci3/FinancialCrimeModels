from squic.SQUIC_functions import *

def load_dataset(name):
    pass

def extract_timeseries(balances):
    pass

def normalization():
    pass

def clustering(X):
    pass

def internal_metrics(X):
    pass

def create_graph(X):
    pass


if __name__ == '__main__':
    # Load dataset (I only pass the name)
    load_dataset()

    # Extract time series
    extract_timeseries()

    # Normalise time series
    normalization()

    # Run SQUIC_fit
    compute_squic()
    
    # Use the extracted W (check if symmetric) for clustering
    clustering()

    # Report internal metrics on the clustering, and visualise with cosmograph
    internal_metrics()
    create_graph()