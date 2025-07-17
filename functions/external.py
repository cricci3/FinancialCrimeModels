import json
import numpy as np
import pandas as pd
from functions.preprocess import PaySim_preprocessing, AMLSim_preprocessing
from functions.SQUIC_functions import *
from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import networkx as nx
from cosmograph import cosmo


def parse_input(user_input):
    """Parse and validate the dataset input in the format NAME_DIMENSION."""
    try:
        name, dimension = user_input.strip().upper().split("_")
    except ValueError:
        raise ValueError("Input must be in the format NAME_DIMENSION (e.g., PAYSIM_10K)")

    valid_names = {"PAYSIM", "AMLSIM"}
    valid_dimensions = {"100", "1K", "10K", "100K", "1M"}

    if name not in valid_names:
        raise ValueError(f"Invalid dataset name '{name}'. Valid options are: {', '.join(valid_names)}")
    if dimension not in valid_dimensions:
        raise ValueError(f"Invalid dimension '{dimension}'. Valid options are: {', '.join(valid_dimensions)}")

    return name, dimension


def load_dataset():
    """Prompt user input and load the corresponding dataset."""
    while True:
        user_input = input("Insert dataset name in the following format NAME_DIMENSION (e.g., PAYSIM_10K): ")

        try:
            name, dimension = parse_input(user_input)
            break
        except ValueError as e:
            print(f"Error: {e}")
            continue

    if name == 'AMLSIM':
        df, account_prop = AMLSim_preprocessing(dimension)
    elif name == 'PAYSIM':
        df, account_prop = PaySim_preprocessing(dimension)

    return df, name, dimension, account_prop


def linear_DA(data_train, labels_train, data_test, labels_test, Theta):
    n_train = labels_train.shape[0]
    n_test = labels_test.shape[0]
    K = np.max(labels_train)  # labels are 1-based (like in R)

    rho = np.zeros((n_test, K))

    print(" entering loop")
    for k in range(1, K+1):
        print(f"    loop n {k}")
        print("     sum")
        # prior probability for class k
        prior = np.sum(labels_train == k) / n_train
        
        print("     mean")
        # sample mean for class k
        mu_k = np.mean(data_train[:, labels_train == k], axis=1)  # mean along samples

        print("     tile")
        # Construct an array by repeating A the number of times given by reps.
        mu_matrix = np.tile(mu_k[:, np.newaxis], (1, n_test))

        print("     adjust")
        # adjusted data
        adjusted_data = data_test - 0.5 * mu_matrix

        # lda score
        print("     lda score")
        lda_score = np.einsum('ij,ji->i', adjusted_data.T, Theta @ mu_matrix) + np.log(prior)
        rho[:, k-1] = lda_score

    predicted = np.argmax(rho, axis=1) + 1  # +1 cause labels start from 1

    f1 = f1_score(labels_test, predicted, average='macro')  # macro = average across classes

    # 0 (no mutual information) and 1 (perfect correlation)
    nmi = normalized_mutual_info_score(labels_test, predicted)

    results_lda = {
        "f1": round(f1, 2),
        "nmi": round(nmi, 2),
    }

    return results_lda


def prepare_LDA(Theta_mtrx, account_prop, target='fraud'):
    ext_metrics = {}

    labels = []

    print("Taking labels of users")
    if target == 'class':
        for user in account_prop.items():
            if user[-1][target] == 'B':
                labels.append('C') # If user is B -> act like it is C (most similar class)
            else:
                labels.append(user[-1][target]) # class can be C, M and B
    elif target == 'fraud':
        for user in account_prop.items():
            labels.append(user[-1][target])

    # Convert to array
    print("Encorder Labels")
    labels = np.array(labels)
    le = LabelEncoder()
    
    labels_encoded = le.fit_transform(labels) + 1  # from C, M to classes 1, 2

    for rho, Theta in Theta_mtrx.items():
        ext_metrics[rho] = {}
        # user_features = Theta.toarray()
        user_features = Theta


        # Split into train/test
        print(f"for rho {rho} train test split")
        data_train, data_test, labels_train, labels_test = train_test_split(
            user_features,
            labels_encoded,
            test_size=0.3,
            random_state=42,
            stratify=labels_encoded
        )

        # Transpose because LDA expects (features x samples)
        data_train = data_train.T  # (features x n_train_samples)
        data_test = data_test.T  

        print(f"for rho {rho} LDA")
        ext_metrics[rho] = linear_DA(data_train, labels_train, data_test, labels_test, Theta)
    
    return ext_metrics


def external_metrics(ext_scores):
    for rho, scores in ext_scores.items():
        print(f"For lambda : {rho}")
        for metric, score in scores.items():
            print(f"    {metric} : {score}")
        print("\n")


def visualize_graph_external(Theta_matrices, account_prop):
    # colors show the clusters
    # intensity of color shows amount of money within this account (node weight)

    widget_dict = {}
    
        
    for l, X in Theta_matrices.items():
        # Create a graph from matrix X
        G = nx.from_numpy_array(X)

        fraud_mapping = {}
        class_mapping = {}

        for node in G.nodes():
            # Check if the node exists in account_prop
            if node in account_prop:
                # print(f"node : {node} in account_prop -> fraud : {account_prop[node]['fraud']}")
                fraud_mapping[node] = account_prop[node]['fraud']
                class_mapping[node] = account_prop[node]['class']
            else:
                # Default value if node not found in account_prop
                fraud_mapping[node] = False
        
        # Set fraud attributes for all nodes
        nx.set_node_attributes(G, fraud_mapping, 'fraud')
        nx.set_node_attributes(G, class_mapping, 'class')

        # Nodes: build dataframe from node attributes
        nodes_data = []
        fraudolent = []

        for node, data in G.nodes(data=True):
            is_fraud = data.get('fraud', False)
            user_class = data.get('class', False)

            # Create conditional label - only show label ID if fraudulent
            label = str(node) if is_fraud else ""

            nodes_data.append({
                'id': node,
                # 'label': str(node),
                'label' : label,
                'color': data.get('color'),
                'degree': G.degree[node],
                'fraud' : is_fraud,
                'class' : user_class
            })
        
        points = pd.DataFrame(nodes_data)

        # Edges: extract links with weights
        links_data = []
        for u, v, d in G.edges(data=True):
            links_data.append({
                'source': u,
                'target': v,
                'weight': d.get('weight', 1.0)
            })
        links = pd.DataFrame(links_data)
        links['weight'] = links['weight'].abs()  # only positive thickness
        links['weight'] = links['weight'] * 10  # only positive thickness

        widget = cosmo(
            points=points,
            links=links,
            point_id_by='id', # id
            point_color_by='class',
            link_source_by='source',
            link_target_by='target',
            link_width_by='weight',
            point_label_by='fraud',
            # point_label_by='id',
            point_size_by='degree',
        )

        widget_dict[l] = widget
    
    return widget_dict
