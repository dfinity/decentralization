import unittest
import random
from tests.test_utils import (
    NetworkData,
    NodeEntry,
    TopologyEntry,
    execute_min_synthetic_nodes_scenario,
)


class TopologyConstraintTests(unittest.TestCase):
    def test_subnet_size_requirement(self):
        topology_entry = (
            TopologyEntry().with_subnet_type("NNS").with_size(13).with_country_limit(13)
        )

        # Create the nodes that should enter the subnet
        nodes = []
        for _ in range(3):
            nodes.append(NodeEntry().with_provider_name("DFINITY").with_country("aa"))

        for _ in range(50):
            nodes.append(NodeEntry().with_country("aa"))

        network_data = (
            NetworkData()
            .with_topology_entry(topology_entry)
            .with_extend_nodes(nodes)
            .build()
        )

        (subnet_changes, status) = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        assert len(subnet_changes) == 1

        subnet = subnet_changes[0]

        assert len(subnet["unchanged"]) == 0
        assert len(subnet["added"]) == 13
        assert len(subnet["removed"]) == 0

    def test_subnet_size_assemble_multiple_subnets(self):
        """
        All of the subnets will require only one country in this test
        This test performs how the tool operates with random subnet sizes.

        Test creates:
          1. NNS with 13 nodes (all can be in the same country)
          2. Between 1 and 25 app subnets (where each can have between 1 and 40 nodes)
        """

        # Generate the NNS config
        nns = (
            TopologyEntry().with_subnet_type("NNS").with_size(13).with_country_limit(13)
        )

        # Generate the random app subnets
        app_subnets = []
        total_nodes_needed_for_app_subnets = 0
        for _ in range(random.randint(1, 25)):
            nodes_needed = random.randint(1, 40)
            app_subnets.append(
                TopologyEntry().with_size(nodes_needed).with_country_limit(nodes_needed)
            )
            # Each app subnet, by default has to have 1 Dfinity owned node
            total_nodes_needed_for_app_subnets += nodes_needed - 1

        nodes_bag = []
        # Generate 3 Dfinity owned nodes + 1 for each other app subnet
        for _ in range(3 + len(app_subnets)):
            nodes_bag.append(
                NodeEntry().with_provider_name("DFINITY").with_country("aa")
            )

        # All other nodes can be arbitrary
        for _ in range(total_nodes_needed_for_app_subnets + nns._subnet_size - 3):
            nodes_bag.append(NodeEntry().with_country("aa"))

        network_data = (
            NetworkData()
            .with_extend_nodes(nodes_bag)
            .with_topology_entry(nns)
            .with_extend_topology_entries(app_subnets)
            .build()
        )

        subnet_changes, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        # Assert that all of the subnets have been filled
        assert len(subnet_changes) == len(app_subnets) + 1

        total_topology = [nns] + app_subnets

        for topology_entry in total_topology:
            resulting_swaps = [
                changes
                for changes in subnet_changes
                if changes["subnet_id"] == topology_entry._subnet_id
            ]

            # The subnet id configuration has to be present in the output
            assert len(resulting_swaps) == 1

            subnet_config = resulting_swaps[0]

            assert len(subnet_config["unchanged"]) == 0
            assert len(subnet_config["removed"]) == 0
            assert len(subnet_config["added"]) == topology_entry._subnet_size

            for added_node in subnet_config["added"]:
                found_node = [
                    node for node in nodes_bag if node._node_id == added_node["node_id"]
                ]
                # Assert that the node has been found in the initial bag
                assert len(found_node) == 1
                found_node = found_node[0]

                nodes_bag.remove(found_node)

        # Assert that all of the nodes have been utilized
        assert len(nodes_bag) == 0

    def test_insufficient_nodes(self):
        nns = (
            TopologyEntry().with_subnet_type("NNS").with_size(13).with_country_limit(13)
        )

        nodes_bag = []
        for _ in range(3):
            nodes_bag.append(
                NodeEntry().with_provider_name("DFINITY").with_country("aa")
            )

        # Anywhere between 1 and 9 will not be enough so it should fail for any of these
        for _ in range(random.randint(1, 9)):
            nodes_bag.append(NodeEntry().with_country("aa"))

        network_data = (
            NetworkData().with_extend_nodes(nodes_bag).with_topology_entry(nns).build()
        )
        _, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status != "Optimal"


if __name__ == "__main__":
    unittest.main()
