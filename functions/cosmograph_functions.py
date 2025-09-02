import networkx as nx
import pandas as pd
from cosmograph import cosmo
import os


COLOR_LIST = [
    ["0", "#4682B4"],  # Steelblue
    ["1", "#FF9999"],  # Red/Pink
    ["2", "#4169E1"],  # Royal Blue
    ["3", "#FFD700"],  # Yellow
    ["4", "#FFCC99"],  # Orange
    ["5", "#CCFFE5"],  # Light blue
    ["6", "#FFFF99"],  # Yellow
    ["7", "#CCFF99"],  # Lime
    ["8", "#CCFFFF"],  # Cyan
    ["9", "#CCCCFF"],  # Lilla
    ["10", "#E9967A"],  # Dark Salmon
    ["11", "#E5CCFF"],  # Dark Lilla
    ["12", "#DC143C"], # Crimson Red
    ["13", "#FFFFFF"], # White
    ["14", "#F0FFF0"],  # Honeydew,
]


def node_to_community(partition):
    # Convert partition list of sets to a node-to-community mapping
    community_mapping = {}
    for community_id, community_set in enumerate(partition):
        for node in community_set:
            community_mapping[node] = str(community_id)

    return community_mapping


def visualize_graph_paysim(W_matrices, squic_method, dict_partition, account_prop, color_by='community', color_list=COLOR_LIST):

    widget_dict = {
        rho: {
        } for rho in W_matrices[squic_method].keys()
    }
    
    for rho, X in W_matrices[squic_method].items():
        # Create a graph from matrix X
        G = nx.from_numpy_array(X)

        community_mapping = node_to_community(dict_partition[squic_method][rho])

        # Sets node attributes from a given value or dictionary of values.
        nx.set_node_attributes(G, community_mapping, 'community')

        class_mapping = {}
        for node in G.nodes():
            if node in account_prop and isinstance(account_prop[node], dict):
                if 'class' in account_prop[node]:
                    class_mapping[node] = account_prop[node]['class']
        
        # Set fraud attributes for all nodes
        nx.set_node_attributes(G, class_mapping, 'class')

        # Nodes: build dataframe from node attributes
        nodes_data = []

        for node, data in G.nodes(data=True):
            user_class = data.get('class', False)

            nodes_data.append({
                'id': node,
                # 'label': str(node),
                'community': data.get('community'),
                'color': data.get('color'),
                'degree': G.degree[node],
                'class' : user_class
            })
        
        points = pd.DataFrame(nodes_data)

        points['community'] = points['community'].astype('category')

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
        
        if color_by == 'community': # 'community' or 'class'
            color_list = COLOR_LIST
        else:
            color_list = [
                ["C", "#4682B4"],  # Steelblue
                ["M", "#FF9999"],  # Red/Pink
            ]

        # COSMOGRAPH docs: https://colab.research.google.com/drive/1Rt8rmmeMuWyFjEqae2DdJ3NYymtjC9cT#scrollTo=IZUK7ioL1xKr
        widget = cosmo(
            points=points,
            links=links,
            link_source_by='source',
            link_target_by='target',
            link_width_by='weight',
            link_color='#E3E3E3',
            link_greyout_opacity=0.1,
            point_color_by=color_by,
            point_size_by='degree',
            point_color_strategy='map',
            point_color_by_map=color_list,
            background_color='#FFFFFF',
        )

        widget_dict[rho] = widget
    
    return widget_dict


def visualize_graph_experiment1(W_matrices, dataset_name, dict_partition, color_by='community', color_list=COLOR_LIST):

    widget_dict = {
        LAMBDA: {
            'louvain' : {},
            'leiden' : {},
            'dbscan' : {},
            'spectral' : {}
        } for LAMBDA in W_matrices[dataset_name].keys()
    }
    
    for LAMBDA, clustering_method in dict_partition.items():
        G = nx.from_numpy_array(W_matrices[dataset_name][LAMBDA])
        # for rho, X in W_matrices[dataset_name].items():
            # Create a graph from matrix X
            # G = nx.from_numpy_array(X)

        for method, partition in clustering_method.items():
            # community_mapping = node_to_community(partition)
            # partition = dict_partition[clustering_method].get(LAMBDA, "Not Found")
            if partition != None:
                community_mapping = node_to_community(partition)

                # Sets node attributes from a given value or dictionary of values.
                nx.set_node_attributes(G, community_mapping, 'community')

                # Nodes: build dataframe from node attributes
                nodes_data = []
                fraudolent = []

                for node, data in G.nodes(data=True):
                    nodes_data.append({
                        'id': node,
                        # 'label': str(node),
                        'community': data.get('community'),
                        'color': data.get('color'),
                        'degree': G.degree[node],
                    })
                
                points = pd.DataFrame(nodes_data)

                points['community'] = points['community'].astype('category')

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
                # links['weight'] = links['weight'] * 10  # only positive thickness

                # COSMOGRAPH docs: https://colab.research.google.com/drive/1Rt8rmmeMuWyFjEqae2DdJ3NYymtjC9cT#scrollTo=IZUK7ioL1xKr
                widget = cosmo(
                    points=points,
                    links=links,
                    link_source_by='source',
                    link_target_by='target',
                    link_width_by='weight', # width of an edge given by weight = value in X between i and j
                    link_color='#E3E3E3',
                    link_greyout_opacity=0.1,
                    # link_color_by='community',
                    point_color_by=color_by,
                    point_size_by='degree',
                    point_color_strategy='map',
                    point_color_by_map=color_list,
                    background_color='#FFFFFF',
                    show_labels_for=fraudolent
                )

                widget_dict[LAMBDA][method] = widget

            else:
                print(f"No partition for rho {LAMBDA}, {clustering_method}")
                widget_dict[LAMBDA][method] = "No clustering"

    return widget_dict


def cosmograph_decomposition(widget_dict, theta, rho, dict_partition, color_list=COLOR_LIST):

    widget_dict[rho] = {
            'louvain' : {},
            'leiden' : {},
            'dbscan' : {},
            'spectral' : {}
    }

    
    for clustering_method, partition in dict_partition.items():
        # Create a graph from matrix X
        G = nx.from_numpy_array(theta)

        if partition != None:
            community_mapping = node_to_community(partition)

            # Sets node attributes from a given value or dictionary of values.
            nx.set_node_attributes(G, community_mapping, 'community')        

            # Nodes: build dataframe from node attributes
            nodes_data = []

            for node, data in G.nodes(data=True):
                nodes_data.append({
                    'id': node,
                    'community': data.get('community'),
                    'color': data.get('color'),
                    'degree': G.degree[node],
                })
            
            points = pd.DataFrame(nodes_data)
            points['community'] = points['community'].astype('category')

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
            # links['weight'] = links['weight'] * 10  # only positive thickness

            # COSMOGRAPH docs: https://colab.research.google.com/drive/1Rt8rmmeMuWyFjEqae2DdJ3NYymtjC9cT#scrollTo=IZUK7ioL1xKr
            widget = cosmo(
                points=points,
                links=links,
                link_source_by='source',
                link_target_by='target',
                link_width_by='weight', # width of an edge given by weight = value in X between i and j
                link_color='#E3E3E3',
                link_greyout_opacity=0.1,
                point_color_by='community',
                point_size_by='degree',
                point_color_strategy='map',
                point_color_by_map=color_list,
                background_color='#FFFFFF',
            )

            widget_dict[rho][clustering_method] = widget

        else:
            print(f"No partition for rho {rho}, {clustering_method}")
            widget_dict[rho][clustering_method] = "No clustering"
    
    return widget_dict


def cosmograph_from_theta(theta, rho, widget_dict):    
    # Create a graph from matrix X
    G = nx.from_scipy_sparse_array(theta)

    nodes_data = []

    for node, data in G.nodes(data=True):
        nodes_data.append({
            'id': node,
            'degree': G.degree[node],
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
    # links['weight'] = links['weight'] * 10  # only positive thickness

    # COSMOGRAPH docs: https://colab.research.google.com/drive/1Rt8rmmeMuWyFjEqae2DdJ3NYymtjC9cT#scrollTo=IZUK7ioL1xKr
    widget = cosmo(
        points=points,
        links=links,
        point_id_by=None,
        link_source_by='source',
        link_target_by='target',
        link_width_by='weight',
        link_color='#E3E3E3',
        link_greyout_opacity=0.1,
        # link_color_by='community',
        # point_color='lightskyblue',
        point_color='#0d5da4',
        point_size_by='degree',
        background_color='#FFFFFF',
    )

    widget_dict[rho] = widget
    
    return widget_dict
