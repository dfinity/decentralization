import unittest
from tests.test_utils import (
    NetworkData,
    NodeEntry,
    TopologyEntry,
    execute_min_synthetic_nodes_scenario,
)


class BlacklistTestScenarios(unittest.TestCase):
    def test_blacklist_node_id(self):
        topology = (
            TopologyEntry()
            .with_subnet_type("NNS")
            .with_size(3)
            .with_subnet_id("subnet")
        )

        blacklisted_node = (
            NodeEntry()
            .with_country("AA")
            .with_provider_name("DFINITY")
            .in_subnet("subnet")
        )

        node_bag = [
            NodeEntry()
            .with_country("BB")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("CC")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
            NodeEntry().with_country("DD").with_provider_name("DFINITY"),
        ]

        network_data = (
            NetworkData()
            .with_topology_entry(topology)
            .with_node_entry(blacklisted_node)
            .with_extend_nodes(node_bag)
            .with_blacklist_entry("node_id", blacklisted_node._node_id)
            .enforce_blacklist()
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        subnet_changes = output[0]

        assert len(subnet_changes["removed"]) == 1

        assert subnet_changes["removed"][0]["node_id"] == blacklisted_node._node_id

    def test_blacklist_node_operator(self):
        topology = (
            TopologyEntry()
            .with_subnet_type("NNS")
            .with_size(3)
            .with_subnet_id("subnet")
        )

        blacklisted_operator_nodes = [
            NodeEntry()
            .with_country("AA")
            .with_operator("op1")
            .in_subnet("subnet")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("AA")
            .with_operator("op1")
            .in_subnet("subnet")
            .with_provider_name("DFINITY"),
        ]

        other_nodes = [
            NodeEntry()
            .with_country("CC")
            .in_subnet("subnet")
            .with_provider_name("DFINITY"),
            NodeEntry().with_country("DD").with_provider_name("DFINITY"),
            NodeEntry().with_country("DD").with_provider_name("DFINITY"),
        ]

        network_data = (
            NetworkData()
            .enforce_blacklist()
            .with_blacklist_entry("node_operator", "op1")
            .with_extend_nodes(blacklisted_operator_nodes)
            .with_extend_nodes(other_nodes)
            .with_topology_entry(topology)
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        subnet_changes = output[0]

        assert len(subnet_changes["removed"]) == 2

        blacklisted_ids = [node._node_id for node in blacklisted_operator_nodes]
        assert all(
            [node["node_id"] in blacklisted_ids for node in subnet_changes["removed"]]
        )

    def test_blacklist_node_provider(self):
        topology = (
            TopologyEntry()
            .with_subnet_type("NNS")
            .with_subnet_id("subnet")
            .with_node_provider_limit(2)
            .with_size(6)
        )

        subnet_nodes = [
            NodeEntry()
            .with_country("AA")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("AA")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("BB")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
            # Non Dfinity nodes
            NodeEntry()
            .with_country("CC")
            .with_provider_name("np1")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("CC")
            .with_provider_name("np2")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("DD")
            .with_provider_name("np3")
            .in_subnet("subnet"),
        ]

        unassigned_nodes = [
            # These ones should be picked
            NodeEntry().with_country("DD").with_provider_name("np3"),
            NodeEntry().with_country("FF").with_provider_name("np4"),
            # These ones should not be picked
            NodeEntry().with_country("EE").with_provider_name("np2"),
            NodeEntry().with_country("EE").with_provider_name("np1"),
        ]

        network_data = (
            NetworkData()
            .with_topology_entry(topology)
            .enforce_blacklist()
            .with_extend_nodes(subnet_nodes)
            .with_extend_nodes(unassigned_nodes)
            .with_blacklist_entry("node_provider", "np1")
            .with_blacklist_entry("node_provider", "np2")
            .build()
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        subnet_changes = output[0]

        assert len(subnet_changes["removed"]) == 2

        assert all(
            [
                node["node_provider"] in ["np1", "np2"]
                for node in subnet_changes["removed"]
            ]
        )

        assert all(
            [
                node["node_provider"] not in ["np1", "np2"]
                for node in subnet_changes["added"]
            ]
        )

    def test_blacklist_data_center(self):
        topology = (
            TopologyEntry()
            .with_subnet_id("subnet")
            .with_subnet_type("NNS")
            .with_size(3)
        )

        nodes_in_subnet = [
            NodeEntry()
            .with_country("AA")
            .with_data_center("dc1")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("BB")
            .with_data_center("dc2")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
            NodeEntry()
            .with_country("CC")
            .with_data_center("dc3")
            .with_provider_name("DFINITY")
            .in_subnet("subnet"),
        ]

        available_nodes = [
            NodeEntry()
            .with_country("AA")
            .with_data_center("dc1")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("BB")
            .with_data_center("dc2")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("CC")
            .with_data_center("dc3")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("DD")
            .with_data_center("dc4")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("FF")
            .with_data_center("dc5")
            .with_provider_name("DFINITY"),
        ]

        network_data = (
            NetworkData()
            .enforce_blacklist()
            .with_blacklist_entry("data_center", "dc1")
            .with_blacklist_entry("data_center", "dc2")
            .with_topology_entry(topology)
            .with_extend_nodes(nodes_in_subnet)
            .with_extend_nodes(available_nodes)
        ).build()

        subnet_changes, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        assert all(
            [
                len(changes) == 2
                for changes in [
                    subnet_changes[0]["added"],
                    subnet_changes[0]["removed"],
                ]
            ]
        )

        expected_removed_nodes = [
            node._node_id for node in nodes_in_subnet if node._dc_id in ["dc1", "dc2"]
        ]
        # Assert that only the nodes with blacklisted dc_id are removed
        assert all(
            [
                removed_node["node_id"] in expected_removed_nodes
                for removed_node in subnet_changes[0]["removed"]
            ]
        )

        possible_added_nodes = [
            node._node_id
            for node in available_nodes
            if node._dc_id not in ["dc1", "dc2"]
        ]
        # Assert that only the nodes with allowed dc_id are added
        assert all(
            [
                added_node["node_id"] in possible_added_nodes
                for added_node in subnet_changes[0]["added"]
            ]
        )

    def test_blacklist_data_center_provider(self):
        topology = (
            TopologyEntry()
            .with_subnet_id("subnet")
            .with_subnet_type("NNS")
            .with_size(3)
        )

        nodes_in_subnet = [
            NodeEntry()
            .with_country("AA")
            .with_provider_name("DFINITY")
            .in_subnet("subnet")
            .with_owner("owner1"),
            NodeEntry()
            .with_country("BB")
            .with_provider_name("DFINITY")
            .in_subnet("subnet")
            .with_owner("owner2"),
            NodeEntry()
            .with_country("CC")
            .with_provider_name("DFINITY")
            .in_subnet("subnet")
            .with_owner("owner3"),
        ]

        available_nodes = [
            NodeEntry()
            .with_country("AA")
            .with_provider_name("DFINITY")
            .with_owner("owner1"),
            NodeEntry()
            .with_country("BB")
            .with_provider_name("DFINITY")
            .with_owner("owner2"),
            NodeEntry()
            .with_country("CC")
            .with_provider_name("DFINITY")
            .with_owner("owner3"),
            NodeEntry()
            .with_country("DD")
            .with_provider_name("DFINITY")
            .with_owner("owner4"),
            NodeEntry()
            .with_country("EE")
            .with_provider_name("DFINITY")
            .with_owner("owner5"),
        ]

        network_data = (
            NetworkData()
            .enforce_blacklist()
            .with_blacklist_entry("data_center_provider", "owner1")
            .with_blacklist_entry("data_center_provider", "owner2")
            .with_topology_entry(topology)
            .with_extend_nodes(nodes_in_subnet)
            .with_extend_nodes(available_nodes)
        ).build()

        subnet_changes, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        assert all(
            [
                len(changes) == 2
                for changes in [
                    subnet_changes[0]["added"],
                    subnet_changes[0]["removed"],
                ]
            ]
        )

        expected_removed_nodes = [
            node._node_id
            for node in nodes_in_subnet
            if node._owner in ["owner1", "owner2"]
        ]
        # Assert that all of the removed nodes have the blacklisted owners
        assert all(
            [
                removed_node["node_id"] in expected_removed_nodes
                for removed_node in subnet_changes[0]["removed"]
            ]
        )

        possible_added_nodes = [
            node._node_id
            for node in available_nodes
            if node._owner not in ["owner1", "owner2"]
        ]
        # Assert that the added nodes don't have the blacklisted owners
        assert all(
            [
                added_node["node_id"] in possible_added_nodes
                for added_node in subnet_changes[0]["added"]
            ]
        )


if __name__ == "__main__":
    unittest.main()
