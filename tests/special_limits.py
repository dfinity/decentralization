"""
This batch of tests is used to verify that the health assertions work
as expected. Only healthy nodes should be used in subnets.
"""

import unittest
from tests.test_utils import (
    NetworkData,
    TopologyEntry,
    NodeEntry,
    execute_min_synthetic_nodes_scenario,
)


class SpecialLimits(unittest.TestCase):
    def test_default_constraints(self):
        subnet = (
            TopologyEntry()
            .with_subnet_id("subnet")
            .with_size(4)
            .with_subnet_type("NNS")
        )

        nodes_in_subnet = [
            NodeEntry()
            .with_country("AA")
            .with_provider_name("DFINITY")
            .with_data_center("sh1")
            .with_owner("Digital Realty")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("BB")
            .with_provider_name("DFINITY")
            .with_data_center("sh1")
            .with_owner("Digital Realty")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("CC")
            .with_provider_name("DFINITY")
            .with_data_center("zh2")
            .with_owner("Everyware")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("DD")
            .with_provider_name("Random")
            .with_data_center("zh2")
            .with_owner("Everyware")
            .in_subnet("subnet"),
        ]

        unassigned_node = (
            NodeEntry()
            .with_country("EE")
            .with_data_center("sh1")
            .with_owner("Everyware")
        )

        network_data = (
            NetworkData()
            .with_topology_entry(subnet)
            .with_extend_nodes(nodes_in_subnet)
            .with_node_entry(unassigned_node)
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        assert len(output) == 1

        subnet_changes = output[0]

        assert len(subnet_changes["added"]) == 0
        assert len(subnet_changes["removed"]) == 0

    def test_dfinity_limit_13_for_nns(self):
        subnet = (
            TopologyEntry()
            .with_subnet_id("subnet")
            .with_subnet_type("NNS")
            .with_size(13)
        )

        six_nodes = []
        for _ in range(6):
            six_nodes.append(
                NodeEntry()
                .with_provider_name("DFINITY")
                .with_country("AA")
                .in_subnet("subnet")
            )

        seven_nodes = []
        for _ in range(7):
            seven_nodes.append(
                NodeEntry().with_provider_name("DFINITY").with_country("AA")
            )

        other = []
        for _ in range(10):
            other.append(NodeEntry().with_country("AA"))

        network_data = (
            NetworkData()
            .with_topology_entry(subnet)
            .with_extend_nodes(six_nodes)
            .with_extend_nodes(seven_nodes)
            .with_extend_nodes(other)
            .with_special_limit("subnet", "node_provider", "DFINITY", 13, "eq")
            .with_special_limit("subnet", "country", "AA", 13, "lt")
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        subnet_change = output[0]
        assert len(subnet_change["unchanged"]) == len(six_nodes)

        assert len(subnet_change["added"]) == len(seven_nodes)

        assert len(subnet_change["removed"]) == 0

        added_providers = list(
            set([node["node_provider"] for node in subnet_change["added"]])
        )

        assert len(added_providers) == 1

        assert added_providers[0] == "DFINITY"


if __name__ == "__main__":
    unittest.main()
