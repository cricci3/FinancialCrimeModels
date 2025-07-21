import networkx as nx
import pandas as pd
from cosmograph import cosmo


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


def visualize_graph_paysim(W_matrices, squic_method, dict_partition, account_prop, name, color_by='community', color_list=COLOR_LIST, show_id=True):

    widget_dict = {
        rho: {
        } for rho in W_matrices[squic_method].keys()
    }
    
    for rho, X in W_matrices[squic_method].items():
        # Create a graph from matrix X
        G = nx.from_numpy_array(X)

        community_mapping = node_to_community(dict_partition[squic_method][rho])
        print(f"For rho {rho}, comm mapping is : {community_mapping}")

        # Sets node attributes from a given value or dictionary of values.
        nx.set_node_attributes(G, community_mapping, 'community')

        fraud_mapping = {}
        class_mapping = {}
        for node in G.nodes():
            if node in account_prop and isinstance(account_prop[node], dict):
                if 'fraud' in account_prop[node]:
                    fraud_mapping[node] = account_prop[node]['fraud']
                if 'class' in account_prop[node]:
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
            if name == 'PAYSIM':
                user_class = data.get('class', False)

            # Create conditional label - only show label ID if fraudulent
            label = str(node) if is_fraud else ""

            if is_fraud:
                fraudolent.append(str(node))

            nodes_data.append({
                'id': node,
                # 'label': str(node),
                'label' : label,
                'community': data.get('community'),
                'color': data.get('color'),
                'degree': G.degree[node],
                'fraud' : is_fraud,
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

        if show_id:
            point_label = 'class' # [id, class or fraud]
        else:
            point_label = None # to produce a graph without labels

        # COSMOGRAPH docs: https://colab.research.google.com/drive/1Rt8rmmeMuWyFjEqae2DdJ3NYymtjC9cT#scrollTo=IZUK7ioL1xKr
        widget = cosmo(
            points=points,
            links=links,
            point_id_by='id',
            link_source_by='source',
            link_target_by='target',
            link_width_by='weight',
            link_color='#E3E3E3',
            link_greyout_opacity=0.1,
            point_color_by=color_by,
            point_label_by=point_label,
            point_size_by='degree',
            point_color_strategy='map',
            point_color_by_map=color_list,
            background_color='#FFFFFF',
            show_labels_for=fraudolent
        )

        widget_dict[rho] = widget
    
    return widget_dict


def visualize_graph_amlsim(W_matrices, squic_method, dict_partition, account_prop, color_by='community', color_list=COLOR_LIST):

    widget_dict = {
        rho: {
            'louvain' : {},
            'dbscan' : {},
            'spectral' : {}
        } for rho in W_matrices[squic_method].keys()
    }
    
    for clustering_method, _ in dict_partition.items():
        for rho, X in W_matrices[squic_method].items():
            # Create a graph from matrix X
            G = nx.from_numpy_array(X)

            # community_mapping = node_to_community(partition)
            community_mapping = node_to_community(dict_partition[clustering_method][rho])

            # Sets node attributes from a given value or dictionary of values.
            nx.set_node_attributes(G, community_mapping, 'community')

            fraud_mapping = {}
            for node in G.nodes():
                if node in account_prop and isinstance(account_prop[node], dict):
                    if 'fraud' in account_prop[node]:
                        fraud_mapping[node] = account_prop[node]['fraud']
                else:
                    # Default value if node not found in account_prop
                    fraud_mapping[node] = False
            
            # Set fraud attributes for all nodes
            nx.set_node_attributes(G, fraud_mapping, 'fraud')

            # Nodes: build dataframe from node attributes
            nodes_data = []
            fraudolent = []

            for node, data in G.nodes(data=True):
                is_fraud = data.get('fraud', False)

                # Create conditional label - only show label ID if fraudulent
                label = str(node) if is_fraud else ""

                if is_fraud:
                    fraudolent.append(str(node))

                nodes_data.append({
                    'id': node,
                    # 'label': str(node),
                    'label' : label,
                    'community': data.get('community'),
                    'color': data.get('color'),
                    'degree': G.degree[node],
                    'fraud' : is_fraud,
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
                point_id_by='id',
                link_source_by='source',
                link_target_by='target',
                link_width_by='weight', # width of an edge given by weight = value in X between i and j
                link_color='#E3E3E3',
                link_greyout_opacity=0.1,
                # link_color_by='community',
                point_color_by=color_by,
                point_label_by='id', # [id, class, fraud]
                point_size_by='degree',
                point_color_strategy='map',
                point_color_by_map=color_list,
                background_color='#FFFFFF',
                show_labels_for=fraudolent
            )

            widget_dict[rho][clustering_method] = widget
    
    return widget_dict


# def visualize_graph(W_matrices, squic_method, account_prop, name, two_comm=True):

#     widget_dict = {
#         rho: {} for rho in W_matrices[squic_method].keys()
#     }
    
        
#     for rho, X in W_matrices[squic_method].items():
#         # Create a graph from matrix X
#         G = nx.from_numpy_array(X)

#         fraud_mapping = {}
#         class_mapping = {}
#         for node in G.nodes():
#             if node in account_prop and isinstance(account_prop[node], dict):
#                 if 'fraud' in account_prop[node]:
#                     fraud_mapping[node] = account_prop[node]['fraud']
#                 if name == 'PAYSIM' and 'class' in account_prop[node]:
#                     class_mapping[node] = account_prop[node]['class']
#             else:
#                 # Default value if node not found in account_prop
#                 fraud_mapping[node] = False
        
#         # Set fraud attributes for all nodes
#         nx.set_node_attributes(G, fraud_mapping, 'fraud')
        
#         if name == 'PAYSIM':
#             nx.set_node_attributes(G, class_mapping, 'class')

#         # Nodes: build dataframe from node attributes
#         nodes_data = []
#         fraudolent = []

#         for node, data in G.nodes(data=True):
#             is_fraud = data.get('fraud', False)
#             if name == 'PAYSIM':
#                 user_class = data.get('class', False)

#             # Create conditional label - only show label ID if fraudulent
#             label = str(node) if is_fraud else ""

#             if is_fraud:
#                 fraudolent.append(str(node))

#             if name != 'PAYSIM':
#                 nodes_data.append({
#                     'id': node,
#                     # 'label': str(node),
#                     'label' : label,
#                     'community': data.get('community'),
#                     'color': data.get('color'),
#                     'degree': G.degree[node],
#                     'fraud' : is_fraud,
#                 })
#             else:
#                 nodes_data.append({
#                 'id': node,
#                 # 'label': str(node),
#                 'label' : label,
#                 'community': data.get('community'),
#                 'color': data.get('color'),
#                 'degree': G.degree[node],
#                 'fraud' : is_fraud,
#                 'class' : user_class
#                 })
        
#         points = pd.DataFrame(nodes_data)

#         points['community'] = points['community'].astype('category')

#         # Edges: extract links with weights
#         links_data = []
#         for u, v, d in G.edges(data=True):
#             links_data.append({
#                 'source': u,
#                 'target': v,
#                 'weight': d.get('weight', 1.0)
#             })
#         links = pd.DataFrame(links_data)
#         links['weight'] = links['weight'].abs()  # only positive thickness
#         # links['weight'] = links['weight'] * 10  # only positive thickness
        
#         if two_comm == True:
#             color_list = [
#                 ["0", "#90ee90"],  # 'Clients': 'mediumseagreen'
#                 ["1", "#00CED1"],  # 'Merchants': 'darkturquoise'
#             ]
#         else:
#             color_list = [
#                 ["0", "#4682B4"],  # Steelblue
#                 ["1", "#FF9999"],  # Red/Pink
#                 ["2", "#4169E1"],  # Royal Blue
#                 ["3", "#FFD700"],  # Yellow
#                 ["4", "#FFCC99"],  # Orange
#                 ["5", "#CCFFE5"],  # Light blue
#                 ["6", "#FFFF99"],  # Yellow
#                 ["7", "#CCFF99"],  # Lime
#                 ["8", "#CCFFFF"],  # Cyan
#                 ["9", "#CCCCFF"],  # Lilla
#                 ["10", "#E9967A"],  # Dark Salmon
#                 ["11", "#E5CCFF"],  # Dark Lilla
#                 ["12", "#DC143C"], # Crimson Red
#                 ["13", "#FFFFFF"], # White
#                 ["14", "#F0FFF0"],  # Honeydew,
#             ]

#         # COSMOGRAPH docs: https://colab.research.google.com/drive/1Rt8rmmeMuWyFjEqae2DdJ3NYymtjC9cT#scrollTo=IZUK7ioL1xKr
#         widget = cosmo(
#             points=points,
#             links=links,
#             point_id_by='id', # id
#             link_source_by='source',
#             link_target_by='target',
#             link_width_by='weight', # width of an edge given by weight = value in X between i and j
#             link_color='#E3E3E3',
#             link_greyout_opacity=0.1,
#             # link_color_by='community',
#             point_color_by='class',
#             # point_label_by='class', # or id or fraud 
#             point_size_by='degree', #'degree
#             point_color_strategy='map',
#             point_color_by_map=color_list,
#             background_color='#FFFFFF',
#             show_labels_for=fraudolent
#         )

#         widget_dict[rho] = widget
    
#     return widget_dict
