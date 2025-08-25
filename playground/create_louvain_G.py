import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms import community
import igraph as ig
import leidenalg as la


def graph_louvain():
    EDGE_COLOR = "lightsteelblue"
    NODE_SIZE = 1500

    # BASIC COLOR
    # COLOR_PALETTE = [
    #     "#E41A1C",  # red
    #     "#377EB8",  # blue
    #     "#4DAF4A",  # green
    #     "#984EA3",  # purple
    #     "#FF7F00",  # orange
    #     "#FFFF33",  # yellow
    #     "#A65628",  # brown
    #     "#F781BF",  # pink
    #     "#999999"   # grey
    # ]

    # More Modern
    COLOR_PALETTE = [
        "#66C2A5",  # teal
        "#FC8D62",  # orange
        "#8DA0CB",  # light blue
        "#E78AC3",  # pink
        "#A6D854",  # lime green
        "#FFD92F",  # yellow
        "#E5C494",  # beige
        "#B3B3B3",  # light grey
        "#8C564B"   # brown
    ]

    # ----- Build a tiny 7-node graph with 3 strong communities -----
    G = nx.Graph()
    # Intra-community edges (strong)
    G.add_edges_from([(0,1),(1,2),(2,0)], weight=3.0)   # C0
    G.add_edge(3,4, weight=2.5)                         # C1
    G.add_edge(5,6, weight=2.5)                         # C2

    # Sparse inter-community links
    G.add_edge(2,3, weight=1)
    G.add_edge(4,5, weight=1)
    G.add_edge(6,0, weight=1)

    # Layout
    pos = nx.spring_layout(G, seed=42, weight="weight")

    # Stage 1: each node its own community
    stage1_colors = {n: i for i, n in enumerate(G.nodes())}

    # Stage 2: Louvain communities
    comms = community.louvain_communities(G, weight="weight", seed=42, resolution=1.0)
    if len(comms) != 3:
        comms = [set([0,1,2]), set([3,4]), set([5,6])]
    node_to_comm = {n: i for i, c in enumerate(comms) for n in c}

    # Stage 3: aggregated graph
    H = nx.Graph()
    H.add_nodes_from(range(len(comms)))
    for u, v, data in G.edges(data=True):
        cu, cv = node_to_comm[u], node_to_comm[v]
        if cu != cv:
            w = data.get("weight", 1.0)
            if H.has_edge(cu, cv):
                H[cu][cv]["weight"] += w
            else:
                H.add_edge(cu, cv, weight=w)

    # Position for aggregated graph
    pos_comm = {cid: np.array([pos[n] for n in comm]).mean(axis=0) for cid, comm in enumerate(comms)}

    # ---- Plot each stage separately ----

    # Stage 1
    plt.figure(figsize=(5,5), dpi=180)
    nx.draw(G, pos,
            node_color=[COLOR_PALETTE[stage1_colors[n] % len(COLOR_PALETTE)] for n in G.nodes()],
            with_labels=False,
            edge_color=EDGE_COLOR,
            width=[G[u][v]["weight"]*1.6 for u,v in G.edges()],
            node_size=NODE_SIZE)
    plt.title("Stage 1: each node = its own community")
    plt.show()

    # Stage 2
    plt.figure(figsize=(5,5), dpi=180)
    nx.draw(G, pos,
            node_color=[COLOR_PALETTE[node_to_comm[n] % len(COLOR_PALETTE)] for n in G.nodes()],
            with_labels=False,
            edge_color=EDGE_COLOR,
            width=[G[u][v]["weight"]*1.6 for u,v in G.edges()],
            node_size=NODE_SIZE)
    plt.title("Stage 2: colored by Louvain community")
    plt.show()

    # Stage 3
    plt.figure(figsize=(5,5), dpi=180)
    nx.draw(H, pos_comm,
            node_color=[COLOR_PALETTE[c % len(COLOR_PALETTE)] for c in range(len(comms))],
            with_labels=False,
            edge_color=EDGE_COLOR,
            width=[H[u][v]["weight"]*2.0 for u,v in H.edges()],
            node_size=NODE_SIZE)
    plt.title("Stage 3: aggregated graph (1 node/community)")
    plt.show()

    print(f"Final number of communities (Stage 3): {len(comms)}")


def graph_dbscan():
    EDGE_COLOR = "lightsteelblue"
    NODE_SIZE = 700

    # Bright distinct colors
    COLOR_PALETTE = [
        "#66C2A5",  # teal
        "#FC8D62",  # orange
        "#8DA0CB",  # light blue
        "#E78AC3",  # pink
        "#A6D854",  # lime green
        "#FFD92F",  # yellow
        "#E5C494",  # beige
        "#B3B3B3",  # light grey
        "#8C564B"   # brown
    ]

    # ----- Build 9-node graph -----
    G = nx.Graph()

    # Cluster of 6 close nodes
    G.add_edges_from([
        (0,1),(1,2),(2,3),(3,4),(4,5),(5,0),  # hexagon-like
        (0,2),(2,4),(4,0)  # extra links inside
    ], weight=2.0)

    # Two nodes near the cluster
    G.add_edge(6, 2, weight=1.5)
    G.add_edge(7, 3, weight=1.5)
    G.add_edge(6, 7, weight=1.0)

    # One far node
    G.add_edge(8, 7, weight=0.5)

    # Layout: initial spring layout
    pos = nx.spring_layout(G, seed=42, weight="weight")

    # Manual tweaks to distances
    pos[1] = pos[1] + np.array([-0.2, 0]) # orange
    pos[2] = pos[2] + np.array([0, 0.2]) # light blue
    pos[3] = pos[3] + np.array([0.22, 0.2]) # pink
    # pos[4] = pos[4] + np.array([0.6, 0.1]) # lime green
    pos[5] = pos[5] + np.array([0.05, -0.3]) # yellow
    pos[6] = pos[6] + np.array([0.2, 0.45]) # beige
    pos[7] = pos[7] + np.array([0.6, 0.45]) # light grey
    pos[8] = pos[8] + np.array([0, 0.5]) # brown

    # ----- Stage 1: each node unique color -----
    stage1_colors = {node: idx for idx, node in enumerate(G.nodes())}

    plt.figure(figsize=(5,5), dpi=180)
    nx.draw(G, pos,
            node_color=[COLOR_PALETTE[stage1_colors[n] % len(COLOR_PALETTE)] for n in G.nodes()],
            with_labels=False,
            edge_color=EDGE_COLOR,
            width=[G[u][v]["weight"]*1.6 for u,v in G.edges()],
            node_size=NODE_SIZE)
    plt.title("Stage 1: each node unique color")
    plt.show()

    # ----- Stage 2: 3 groups -> colors -----
    group1 = {0,1,2,3,4,5}  # main cluster
    group2 = {6,7}          # near nodes
    group3 = {8}            # far node

    node_to_comm_stage2 = {}
    for n in group1:
        node_to_comm_stage2[n] = 0
    for n in group2:
        node_to_comm_stage2[n] = 1
    for n in group3:
        node_to_comm_stage2[n] = 2

    plt.figure(figsize=(5,5), dpi=180)
    nx.draw(G, pos,
            node_color=[COLOR_PALETTE[node_to_comm_stage2[n] % len(COLOR_PALETTE)] for n in G.nodes()],
            with_labels=False,
            edge_color=EDGE_COLOR,
            width=[G[u][v]["weight"]*1.6 for u,v in G.edges()],
            node_size=NODE_SIZE)
    plt.title("Stage 2: 6 close / 2 near / 1 far")
    plt.show()

    # ----- Stage 3: merge near nodes into main cluster -----
    node_to_comm_stage3 = {}
    for n in group1 | group2:  # merged cluster
        node_to_comm_stage3[n] = 0
    for n in group3:
        node_to_comm_stage3[n] = 1

    plt.figure(figsize=(5,5), dpi=180)
    nx.draw(G, pos,
            node_color=[COLOR_PALETTE[node_to_comm_stage3[n] % len(COLOR_PALETTE)] for n in G.nodes()],
            with_labels=False,
            edge_color=EDGE_COLOR,
            width=[G[u][v]["weight"]*1.6 for u,v in G.edges()],
            node_size=NODE_SIZE)
    plt.title("Stage 3: merged near nodes with cluster")
    plt.show()


if __name__ == '__main__':
    SELECTED = 'LOUVAIN'

    if SELECTED == 'DBSCAN':
        graph_louvain()
    else:
        graph_dbscan()
