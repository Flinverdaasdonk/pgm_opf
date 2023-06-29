"""
A collection of utility functions to work with graph in the power-grid-model graph format.

Based on sources:
[1] Convex Relaxations of Optimal Power Flow, Part I. By Steven Low, https://arxiv.org/abs/1405.0766
[2] Towards Distributed Energy Services: Decentralizing Optimal Power Flow with Machine Learning, By Roel Dobbe, https://arxiv.org/pdf/1806.06790.pdf

date: 9-6-2023
Author: Flin Verdaasdonk
"""

import numpy as np


def add_incoming_lines(network):
    """
    determine which lines are 'before' each line. 
    log the indices of those lines
    """

    line = network["line"]
    line["incoming_line_indices"] = []


    for from_node in line["from_node"]:
        current_incoming_lines = [] # use this to keep track of incoming lines

        # incoming lines have the same 'to_node' as from node
        for j, to_node in enumerate(line["to_node"]):
            if from_node == to_node:
                current_incoming_lines.append(j)
        
        line["incoming_line_indices"].append(current_incoming_lines)
    network["line"] = line
    return network

def add_outgoing_lines(network):
    # determine which lines are 'after' each line
    # log the indices of those lines

    line = network["line"]
    line["outgoing_line_indices"] = []

    for to_node in line["to_node"]:
        current_outgoing_lines = [] # use this to keep track of incoming lines

        # incoming lines have the same 'to_node' as from node
        for j, from_node in enumerate(line["from_node"]):
            if to_node == from_node:
                current_outgoing_lines.append(j)
        
        line["outgoing_line_indices"].append(current_outgoing_lines)
    
    network["line"] = line
    return network


def id_to_index(component, id):
    """
    Get the index (which element it is in the list) of the current id, for a given component
    """

    return component["id"].index(id)

def index_to_id(component, idx):
    """
    Get the id of the current index, for a given component
    """
        
    return component["id"][idx]

def add_downstream_nodes(network):
    """
    For all nodes, indicate which nodes are downstream to all nodes. 
    Only considers the first-order children (so A --->---B --->--- C will only list B as a downstream node for A)

    eg. node["downstream_node_ids"] will become a list where each element corresponds to the node with the same index, 
    and each value will be another list, that contains the IDs of child nodes
    """
        
    node = network["node"]
    line = network["line"]

    node["downstream_node_indices"] = []
    node["downstream_node_ids"] = []

    for current_node_id in node["id"]:
        per_node_downstream_node_ids = []
        per_node_downstream_node_indices = []

        for line_from_node_id, line_to_node_id in zip(line["from_node"], line["to_node"]):
            if current_node_id == line_from_node_id:
                per_node_downstream_node_ids.append(line_to_node_id)
                line_to_node_idx = id_to_index(node, line_to_node_id)
                per_node_downstream_node_indices.append(line_to_node_idx)

        node["downstream_node_ids"].append(per_node_downstream_node_ids)
        node["downstream_node_indices"].append(per_node_downstream_node_indices)
    
    network["node"] = node
    network["line"] = line
    return network
        
def add_upstream_nodes(network):
    """
    For all nodes, indicate which nodes are upstream to all nodes. 
    Only considers the first-order parents (so A --->---B --->--- C will only list B as a upstream node for C)

    eg. node["upstream_node_ids"] will become a list where each element corresponds to the node with the same index, 
    and each value will be another list, that contains the IDs of parent nodes
    """
        
    node = network["node"]
    line = network["line"]

    node["upstream_node_indices"] = []
    node["upstream_node_ids"] = []

    for current_node_id in node["id"]:
        per_node_upstream_node_ids = []
        per_node_upstream_node_indices = []

        for line_from_node_id, line_to_node_id in zip(line["from_node"], line["to_node"]):
            if current_node_id == line_to_node_id:
                per_node_upstream_node_ids.append(line_from_node_id)
                line_from_node_idx = id_to_index(node, line_from_node_id)
                per_node_upstream_node_indices.append(line_from_node_idx)

        node["upstream_node_ids"].append(per_node_upstream_node_ids)
        node["upstream_node_indices"].append(per_node_upstream_node_indices)
    
    network["node"] = node
    network["line"] = line
    return network


def node_ids_to_line_index(upstream_node, downstream_node, line):
    """
    Given two node ids, this function will return the index of the line that connects them
    """
    
    # iterate over the from_nodes/to_nodes in line
    for line_index, (from_node, to_node) in enumerate(zip(line["from_node"], line["to_node"])):

        # if the from_node is the upstream_node in the arguments AND to_node is the downstream node, it means we found a connecting line
        if from_node == upstream_node and to_node == downstream_node:
            return line_index

    raise Exception(f"No line index found with downstream_node id={downstream_node} and upstream_node id={upstream_node}")


def add_power_keys(node):
    """
    Used to quickly add keys to the node dictionary that store the power info 
    (this info is initially stored in the source/load data, but these uniquely map to a node
    """

    # If it hasn't been added before
    if not "p_min" in list(node.keys()):

        node["p_min"] = []
        node["p"] = []
        node["p_max"] = []

        node["q_min"] = []
        node["q"] = []
        node["q_max"] = []
    return node

def match_sources_to_nodes(network):
    """
    Add the source information (p_min/p_max, etc.) to the corresponding nodes
    """

    source = network["source"]
    node = network["node"]

    node = add_power_keys(node)
    for i in range(len(source["node"])):
        for key in ["p_min", "p", "p_max", "q_min", "q", "q_max"]:

            node[key].append(source[key][i])
    
    network["node"] = node
    return network
        

def match_loads_to_nodes(network):
    """
    Add the load information (p_min/p_max, etc.) to the corresponding nodes
    """
        
    sym_load = network["sym_load"]
    node = network["node"]

    node = add_power_keys(node)
    for i in range(len(sym_load["node"])):
        for key in ["p_min", "p", "p_max", "q_min", "q", "q_max"]:
            node[key].append(sym_load[key][i])

    network["node"] = node
    return network

def make_simple_radial_net():
    """
    returns a net with the following topology

node_1 ---line_3--- node_2 ----line_5---- node_6
|                    |                     |
source_10          sym_load_4           sym_load_7

    """
    node = {}
    node["id"] = [1, 2, 6]
    node["v_max"] = [1.1, 1.1, 1.1]
    node["v_min"] = [0.9, 0.9, 0.9]

    line = {}
    line["id"] = [3, 5]
    line["from_node"] = [1, 2]
    line["to_node"] = [2, 6]
    line["r1"] = [0.25, 0.25]
    line["x1"] = [0.2, 0.2]
    line["i_max"] = [1_000, 1_000]

    sym_load = {}
    sym_load["id"] = [4, 7]
    sym_load["node"] = [2, 6]

    sym_load["p_min"] = [0, 0]
    sym_load["p"] = [500, 700]
    sym_load["p_max"] = [1000, 2000]

    sym_load["q_min"] = [0, 0]
    sym_load["q"] = [0, 10]
    sym_load["q_max"] = [100, 200]


    source = {}
    source["id"] = [10]
    source["node"] = [1]
    source["p_min"] = [0]
    source["p"] = [5000]
    source["p_max"] = [10_000]

    source["q_min"] = [-5_000]
    source["q"] = [0]
    source["q_max"] = [5_000]

    # compiling the results
    network = {"node": node, "line": line, "sym_load": sym_load, "source": source}

    # add additional graph data.
    network = add_upstream_nodes(network)
    network = add_downstream_nodes(network)
    network = match_sources_to_nodes(network)
    network = match_loads_to_nodes(network)
    
    network = add_incoming_lines(network)
    network = add_outgoing_lines(network)
    network = add_line_impedances(network)
    network = add_line_admittances(network)

    return network

def make_simple_mesh_net():
    """
    returns a net with the following topology
 -------------------- line_8-----------------
 |                                          |
node_1 ---line_3--- node_2 ----line_5---- node_6
 |                    |                     |
source_10          sym_load_4           sym_load_7

    """

    node = {}
    node["id"] = [1, 2, 6]
    node["v_max"] = [1.1, 1.1, 1.1]
    node["v_min"] = [0.9, 0.9, 0.9]

    line = {}
    line["id"] = [3, 5, 8]
    line["from_node"] = [1, 2, 1]
    line["to_node"] = [2, 6, 6]
    line["r1"] = [0.25, 0.25, 0.25]
    line["x1"] = [0.2, 0.2, 0.2]
    line["i_max"] = [1_000, 1_000, 1_000]

    sym_load = {}
    sym_load["id"] = [4, 7] # load id
    sym_load["node"] = [2, 6] # to which node it is connected

    sym_load["p_min"] = [0, 0] # minimum real power 
    sym_load["p"] = [500, 700] # real power at the present time
    sym_load["p_max"] = [1000, 2000] # max real power 

    sym_load["q_min"] = [0, 0] # minimum reactive power
    sym_load["q"] = [50, 0] # reactive power at the present time
    sym_load["q_max"] = [100, 200] #max reactive power 

    source = {}
    source["id"] = [10] # source id
    source["node"] = [1] # to which node the source is connected

    source["p_min"] = [-5000] # 
    source["p"] = [5_000] # 
    source["p_max"] = [10_000]

    source["q_min"] = [-5_000]
    source["q"] = [5_000] # 
    source["q_max"] = [5_000]

    # compiling the results
    network = {"node": node, "line": line, "sym_load": sym_load, "source": source}

    # add additional graph data.
    network = add_upstream_nodes(network)
    network = add_downstream_nodes(network)
    network = match_sources_to_nodes(network)
    network = match_loads_to_nodes(network)
    
    network = add_incoming_lines(network)
    network = add_outgoing_lines(network)
    network = add_max_line_powers(network)
    network = add_line_impedances(network)
    network = add_line_admittances(network)

    return network

def add_max_line_powers(network):
    line = network["line"]
    line["p_max"] = [r1*i_max**2 for r1, i_max in zip(line["r1"], line["i_max"])]

    network["line"] = line
    return network


def node_ids_to_line_admittance(upstream_node_id, downstream_node_id, network):
    """
    using the ids of two connecting nodes, get the corresponding line admittance
    """
    line = network["line"]

    line_idx = node_ids_to_line_index(upstream_node=upstream_node_id, downstream_node=downstream_node_id, line=line)
    
    # get impedance
    z = line["z1"][line_idx]
    
    # get admittance
    y = 1/z
    
    return y

def add_line_impedances(network):
    """
    For each line, calculate the line impedance from the resistance/reactance
    """

    line = network["line"]
    line["z1"] = []

    # for all lines
    for i in range(len(line["id"])):
        # get the corresponding resistance
        R = line["r1"][i]

        # get the corresponding reactance
        X = line["x1"][i]

        # calculate impedance
        Z = complex(R, X)

        # add impedance to the impedance list
        line["z1"].append(Z)


    network["line"] = line
    return network

def add_line_admittances(network):
    """
    For each line, calculate the line admittacne from the impedance
    """
        
    line = network["line"]

    # If the impedance hasn't been calculated yet
    if not ["z1"] in list(line.keys()):

        # add the impedance
        network = add_line_impedances(network)
        line = network["line"]
    
    # set the admittance for each line to 1/impedanc2
    line["y1"] = [1/z for z in line["z1"]]

    network["line"] = line
    return network

def make_admittance_matrix(network):
    """ The admittance matrix is an (n+1)*(n+1) matrix (where n = number of nodes, including bus node). 
    Diagonal entries are the sum of all the admittances of the cables connecting to node i (Y_ij = \sum_{k:k~i}y_ik if i=j)
    Off-diagonal entries are:
        - minus the line admittance if nodes i and j are connected: Y_ij = -y_ij if i!=j and i~j
        - zero otherwise

    See page 10 of source [1] 
    """

    N = len(network["node"]["id"])
    node = network["node"]
    line = network["line"]

    # initialize complex zero matrix
    Y = np.zeros([N, N], dtype=np.cdouble)

    # for all rows
    for r in range(N):
        # for all columns
        for c in range(N):

            # get the node ids
            r_id = node["id"][r]
            c_id = node["id"][c]

            # if the row equals the column, we're at a diagonal
            if r == c:
                
                # initialize admittance
                admittance = 0

                # add the admittances of all downstream nodes
                admittance += sum([node_ids_to_line_admittance(upstream_node_id=r_id, downstream_node_id=dn_id, network=network) for dn_id in node["downstream_node_ids"][r]])
                
                # add the admittance of all upstream nodes
                admittance += sum([node_ids_to_line_admittance(upstream_node_id=un_id, downstream_node_id=r_id, network=network) for un_id in node["upstream_node_ids"][r]])
                Y[r, c] = admittance

            # if we're not on the diagonal
            else:

                # if the column node  is one of the downstream nodes of the current row node
                if c_id in node["downstream_node_ids"][r]:
                    Y[r, c] = -node_ids_to_line_admittance(upstream_node_id=r_id, downstream_node_id=c_id, network=network)
                
                # if the column node  is one of the upstream nodes of the current row node
                elif c_id in node["upstream_node_ids"][r]: 
                    Y[r, c] = -node_ids_to_line_admittance(upstream_node_id=c_id, downstream_node_id=r_id, network=network)
                
                # if the nodes aren't connected, we leave the admittance at zero
                else:
                    pass    

    return Y

def make_ej(n_nodes, j):
    """
    Vector of size 'n_nodes', with all zero's except for the jth element

    See page 10 source [1]
    """
    ej = np.zeros(n_nodes)
    ej[j] = 1
    return ej

def make_Jj(Y, j):
    """
    Matrix of same size as the admittance matrix Y, with all zero's except for the (j, j)th element

    See page 10 source [1]
    """

    Jj = np.zeros(Y.shape)
    Jj[j, j] = 1
    return Jj

def make_Yj(Y, j):
    """
    Matrix of same size as the admittance matrix Y, with all zero's except for the jth row, 
    which contains the same elements as the jth row of Y

    See page 10 source [1]
    """
    Yj = np.zeros_like(Y)
    Yj[j, :] = Y[j, :]
    return Yj

def make_Phij(Y, j):
    """
    Corresponds somewhat to the real version of the Yj matrix (note, it is a complex matrix, so not solely real)
    
    # page 10 source [1]
    """

    
    Yj = make_Yj(Y, j)
    Psij = (np.conj(Yj.T) + Yj)/2
    return Psij
 
def make_Psij(Y, j):
    """
    Corresponds somewhat to the imag version of the Yj matrix (note, it is a complex matrix, so not solely imaginary/real)
    
    # page 10 source [1]
    """

    Yj = make_Yj(Y, j)
    Psij = (np.conj(Yj.T)- Yj)/complex(0, 2)
    return Psij