"""
This batch of tests is used to verify that the health assertions work
as expected. Only healthy nodes should be used in subnets.
"""

import unittest

from tests.test_utils import (
    NetworkData,
    NodeEntry,
    TopologyEntry,
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

    def test_respect_default_limit(self):
        subnet_1 = (
            TopologyEntry()
            .with_subnet_id("subnet_1")
            .with_subnet_type("Application")
            .with_country_limit(3)
            .with_size(5)
        )
        subnet_2 = (
            TopologyEntry()
            .with_subnet_id("subnet_2")
            .with_subnet_type("Application")
            .with_country_limit(3)
            .with_size(5)
        )

        aa_nodes = []
        bb_nodes = []
        cc_nodes = []

        for _ in range(10):
            aa_nodes.append(NodeEntry().with_country("AA"))
            bb_nodes.append(NodeEntry().with_country("BB"))
            cc_nodes.append(NodeEntry().with_country("CC"))

        network_data = (
            NetworkData()
            .with_topology_entry(subnet_1)
            .with_topology_entry(subnet_2)
            .with_extend_nodes(aa_nodes)
            .with_extend_nodes(bb_nodes)
            .with_extend_nodes(cc_nodes)
            .with_special_limit("default", "country", "AA", 2, "eq")
            .with_special_limit("default", "country", "BB", 3, "eq")
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        aa_nodes_ids = [node._node_id for node in aa_nodes]
        bb_nodes_ids = [node._node_id for node in bb_nodes]
        cc_nodes_ids = [node._node_id for node in cc_nodes]

        for subnet in output:
            assert len(subnet["added"]) == 5
            assert len(subnet["removed"]) == 0

            added_nodes = [node["node_id"] for node in subnet["added"]]
            # Out of added nodes only 2 are in aa
            assert len([node for node in added_nodes if node in aa_nodes_ids]) == 2

            # Out of added nodes only 3 are in bb
            assert len([node for node in added_nodes if node in bb_nodes_ids]) == 3

            # Out of added nodes 0 are in cc
            assert len([node for node in added_nodes if node in cc_nodes_ids]) == 0

    def test_append_from_default(self):
        subnet = (
            TopologyEntry()
            .with_subnet_id("subnet")
            .with_size(10)
            .with_subnet_type("Application")
            .with_country_limit(5)
        )

        specific_np_nodes_aa = []
        for i in range(5):
            specific_np_nodes_aa.append(
                NodeEntry()
                .with_provider_name("specific_np")
                .with_country(f"country_{i}")
            )

        aa_country_nodes = []
        for _ in range(5):
            aa_country_nodes.append(NodeEntry().with_country("AA"))

        other_nodes = []
        for i in range(10):
            other_nodes.append(NodeEntry().with_country("other_country_" + str(i)))

        network_data = (
            NetworkData()
            .with_topology_entry(subnet)
            .with_extend_nodes(specific_np_nodes_aa)
            .with_extend_nodes(aa_country_nodes)
            .with_extend_nodes(other_nodes)
            .with_special_limit("subnet", "node_provider", "specific_np", 5, "eq")
            .with_special_limit("default", "country", "AA", 5, "eq")
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        subnet_change = output[0]

        added_ids = [node["node_id"] for node in subnet_change["added"]]

        assert len(added_ids) == 10

        specific_np_nodes_aa_ids = [node._node_id for node in specific_np_nodes_aa]
        aa_country_nodes_ids = [node._node_id for node in aa_country_nodes]
        # The ids used should be either aa country or specific_np
        for id in added_ids:
            assert id in specific_np_nodes_aa_ids or id in aa_country_nodes_ids

    def test_dont_override_specific(self):
        subnet = (
            TopologyEntry()
            .with_subnet_id("subnet")
            .with_subnet_type("Application")
            .with_size(3)
            .with_country_limit(3)
        )

        nodes = []
        for _ in range(2):
            nodes.append(NodeEntry().with_country("AA"))
            nodes.append(NodeEntry().with_country("BB"))

        network_data = (
            NetworkData()
            .with_topology_entry(subnet)
            .with_extend_nodes(nodes)
            .with_special_limit("subnet", "country", "AA", 1, "eq")
            .with_special_limit("subnet", "country", "BB", 2, "eq")
            .with_special_limit("default", "country", "AA", 2, "eq")
            .with_special_limit("default", "country", "BB", 1, "eq")
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        added_nodes = [node["node_id"] for node in output[0]["added"]]
        assert len(added_nodes) == 3

        # Expect to have 1 AA and 2 BB nodes
        nodes_aa = [node._node_id for node in nodes if node._country == "AA"]
        nodes_bb = [node._node_id for node in nodes if node._country == "BB"]

        aa_added_nodes = 0
        bb_added_nodes = 0
        for node in added_nodes:
            if node in nodes_aa:
                aa_added_nodes += 1
            elif node in nodes_bb:
                bb_added_nodes += 1
            else:
                raise ValueError("Node is not in any seeded country")

        assert aa_added_nodes == 1
        assert bb_added_nodes == 2


if __name__ == "__main__":
    unittest.main()
