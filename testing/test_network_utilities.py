import unittest

from utils import network_utilities as nut

class TestNetworkUtilities(unittest.TestCase):
    """
    Test all functions in 'add_utility_graph_data()' using the network
    in 'make_net()'. For reference, this network looks like this:
        
    n1(S16)-----Line8-------- n2(S17)
    |                        |
    Line9                  Line10
    |                        |
    n3(L18)---Line11--------n4(L19)-------------
    |                        |                  |
    Line12                 Line13              Line14
    |                        |                  |
    n5(L20)---Line15------n6(L21)----Line16---n7(S22)

    
     
    """
    def __init__(self, *args, **kwargs):
        super(TestNetworkUtilities, self).__init__(*args, **kwargs)
        self.network = nut.make_net()

    def test_add_upstream_nodes(self):
        """ Test if the 'add_upstream_nodes' function does what we expect it to do
        """
        func = self.network["node"]["upstream_node_ids"]
        manual = [[],[1],[1],[2, 3],[3],[4, 5],[6, 4]]

        # make them sets since the specific order of the upstream nodes doesn't matter
        func_set = [set(x) for x in func]
        manual_set = [set(x) for x in manual]

        self.assertEqual(func_set, manual_set)

    def test_add_downstream_nodes(self):
        """ Test if the 'down_upstream_nodes' function does what we expect it to do
        """
        func = self.network["node"]["downstream_node_ids"]
        manual = [[2, 3], [4], [4, 5], [6, 7], [6], [7], []]

        # make them sets since the specific order of the upstream nodes doesn't matter
        func_set = [set(x) for x in func]
        manual_set = [set(x) for x in manual]

        self.assertEqual(func_set, manual_set)

    
    def test_add_incoming_lines(self):
        """ Test if the 'add_incoming_lines' function does what we expect it to do
        """

        # in practice we'll use the incoming_line_INDICES
        func = self.network["line"]["incoming_line_indices"]

        # for verification we start with the IDs since this is more intuitive
        manual_ids = [[], [], [8], [9], [9], [10, 11], [10, 11], [12],[13, 15]]

        # convert these IDs to indices
        manual_indices = []
        for id_ls in manual_ids:
            idx_ls = [nut.id_to_index(self.network["line"], id) for id in id_ls]
            manual_indices.append(idx_ls)

        # make them sets since the specific order of the upstream nodes doesn't matter
        func_set = [set(x) for x in func]
        manual_set = [set(x) for x in manual_indices]

        self.assertEqual(func_set, manual_set)

    def test_add_outgoing_lines(self):
        """ Test if the 'add_outgoing_lines' function does what we expect it to do
        """

        # in practice we'll use the outgoing_line_INDICES
        func = self.network["line"]["outgoing_line_indices"]

        # for verification we start with the IDs since this is more intuitive to build from the graph structure
        manual_ids = [[10], [11, 12], [13, 14], [13, 14], [15], [16], [], [16], []]

        # convert these IDs to indices
        manual_indices = []
        for id_ls in manual_ids:
            idx_ls = [nut.id_to_index(self.network["line"], id) for id in id_ls]
            manual_indices.append(idx_ls)

        # make them sets since the specific order of the upstream nodes doesn't matter
        func_set = [set(x) for x in func]
        manual_set = [set(x) for x in manual_indices]

        self.assertEqual(func_set, manual_set)


if __name__ == "__main__":
    unittest.main()
