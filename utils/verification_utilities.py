from copy import deepcopy

def verify_network(network):
    verify_component_keys(network)
    verify_component_number_of_elements(network)
    verify_component_element_values(network)

    verify_if_all_nodes_connected(network)
    verify_if_all_lines_connected(network)
    verify_for_unique_ids(network)

def verify_for_unique_ids(network):
    id_list = []
    components = ["node", "line", "sym_load", "source"]

    for component in components:
        id_list += network[component]["id"]

    id_set = set(id_list)

    assert len(id_set) == len(id_list), f"verify_for_unique_ids: length between id list and set is not the same, which implies duplicates in list.\n list={id_list} \n set={id_set}"

def verify_component_keys(network):
    verify_node_keys(network)
    verify_line_keys(network)
    verify_source_keys(network)
    verify_sym_load_keys(network)

def verify_node_keys(network):
    component = "node"
    keys = ["id", "u_rated", "u_min", "u_max"]

    for k in keys:
        assert k in list(network[component].keys()), f"verify_{component}_keys: key {k} not in {network[component].keys()}"


def verify_line_keys(network):
    component = "line"
    keys = ["id", "from_node", "to_node", "from_status", "to_status", "r1", "x1", "c1", "tan1", "i_n",  "i_max"]

    for k in keys:
        assert k in list(network[component].keys()), f"verify_{component}_keys: key {k} not in {network[component].keys()}"

def verify_source_keys(network):
    component = "source"
    keys = ["id", "node", "status", "u_ref", "p_min", "p_specified", "p_max", "q_min", "q_specified", "q_max"]
    
    for k in keys:
        assert k in list(network[component].keys()), f"verify_{component}_keys: key {k} not in {network[component].keys()}"

def verify_sym_load_keys(network):
    component = "sym_load"
    keys = ["id", "node", "status", "type", "p_min", "p_specified", "p_max", "q_min", "q_specified", "q_max"]
    
    for k in keys:
        assert k in list(network[component].keys()), f"verify_{component}_keys: key {k} not in {network[component].keys()}"

def verify_component_number_of_elements(network):
    components = ["node", "line", "sym_load", "source"]

    for component in components:
        n_elements = len(network[component]["id"])
        # print(component, list(network[component].keys()))
        for key in list(network[component].keys()):
            assert n_elements == len(network[component][key]), f"verify_component_number_of_elements: for {component}, len(id) (={n_elements}) != len({key}) (={len(network[component][key])})"

def verify_component_element_values(network):
    verify_node_element_values(network)
    verify_line_element_values(network)
    verify_sym_load_element_values(network)
    verify_source_element_values(network)

def verify_node_element_values(network):
    pass

def verify_line_element_values(network):
    pass

def verify_sym_load_element_values(network):
    component = "sym_load"
    N = len(network[component]["id"])

    for i in range(N):
        assert network[component]["p_min"][i] <= network[component]["p_specified"][i] <= network[component]["p_max"][i], f"verify_sym_load_element_values: failed at checking p_specified at index {i}"
        assert network[component]["q_min"][i] <= network[component]["q_specified"][i] <= network[component]["q_max"][i], f"verify_sym_load_element_values: failed at checking q_specified at index {i}"


def verify_source_element_values(network):
    component = "source"
    N = len(network[component]["id"])

    for i in range(N):
        assert network[component]["p_min"][i] <= network[component]["p_specified"][i] <= network[component]["p_max"][i], f"verify_source_element_values: failed at checking p_specified at index {i}"
        assert network[component]["q_min"][i] <= network[component]["q_specified"][i] <= network[component]["q_max"][i], f"verify_source_element_values: failed at checking q_specified at index {i}"

def verify_if_all_nodes_connected(network):
    node_ids = deepcopy(network["node"]["id"])

    for from_node, to_node in zip(network["line"]["from_node"], network["line"]["to_node"]):
        if from_node in node_ids:
            node_ids.remove(from_node)

        if to_node in node_ids:
            node_ids.remove(to_node)
            
    assert len(node_ids) == 0, f"verify_if_all_nodes_connected: these nodes are not connected: {node_ids}"                                                                                                                                                                                             


def verify_if_all_lines_connected(network):
    node_ids = network["node"]["id"]

    for i, (from_node, to_node) in enumerate(zip(network["line"]["from_node"], network["line"]["to_node"])):
        assert from_node in node_ids, f"verify_if_all_lines_connected: line at idx {i} has non existant from_node (id={from_node})"
        assert to_node in node_ids, f"verify_if_all_lines_connected: line at idx {i} has non existant to_node (id={to_node})"