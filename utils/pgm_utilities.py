from power_grid_model import LoadGenType
from power_grid_model import PowerGridModel, CalculationMethod, CalculationType
from power_grid_model import initialize_array
from power_grid_model.validation import assert_valid_input_data
import numpy as np

def network_to_pgm_format(network, validate_pgm_input=True):
    # these keys should be taken from the network data in opf format, and added to the pgm input_data
    relevant_keys = {"node": ['id', "u_rated"],
        "line": ["id", "from_node", "to_node", "from_status", "to_status", "r1", "x1", "c1", "tan1", "i_n"],
        "sym_load": ["id", "node", "status", "type", "p_specified", "q_specified"],
        "source": ["id", "node", "status", "u_ref"]}
    

    # At the moment, the opf-network format contains the load type as str instead of pgm datatype; fix this
    for idx, load_type in enumerate(network["sym_load"]["type"]):
        if load_type == "LoadGenType.const_power":
            network["sym_load"]["type"][idx] = LoadGenType.const_power

        else:
            raise Exception(f"For idx={idx}, load type = {load_type} not found")
    
    #
    pgm_input_data = {}

    for component, keys in relevant_keys.items():
        pgm_input_data[component] = initialize_array("input", component, len(network[component]["id"]))
        for key in keys:
            pgm_input_data[component][key] = network[component][key]

    if validate_pgm_input:
        assert_valid_input_data(input_data=pgm_input_data, calculation_type=CalculationType.power_flow)

    return pgm_input_data

def quantify_congestion(pgm_output, network):
    """
    Quantify the congestion given the presented pgm_output and networks
    """
    line_limits = network["line"]["i_max"]
    line_currents = pgm_output["line"]["i_from"] # CHECK IF I_FROM IS THE CORRECT ONE; WHY NOT I_TO?
    
    line_congestion_func = lambda i, i_max : np.max([0, np.abs(i)-i_max])**2

    net_congestion = np.sum([line_congestion_func(i=i, i_max=i_max) for i, i_max in zip(line_currents, line_limits)])

    return net_congestion


if __name__ == "__main__":
    pass