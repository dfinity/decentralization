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


class TestHealthConstraints(unittest.TestCase):
    def test_remove_dead_node(self):
        healthy_nodes = [
            NodeEntry().with_country("AA").with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("BB")
            .with_provider_name("DFINITY")
            .with_status("UNASSIGNED"),
            NodeEntry()
            .with_country("CC")
            .with_provider_name("DFINITY")
            .with_status("DOWN"),
            NodeEntry()
            .with_country("DD")
            .with_provider_name("DFINITY")
            .with_status("DEGRADED"),
        ]
        unhealthy_node = (
            NodeEntry().with_country("CC").with_status("DOWN").in_subnet("subnet")
        )
        network_data = (
            NetworkData()
            .enforce_health()
            .with_topology_entry(
                TopologyEntry()
                .with_subnet_id("subnet")
                .with_size(3)
                .with_subnet_type("NNS")
            )
            .with_node_entry(
                NodeEntry()
                .with_country("AA")
                .with_status("UP")
                .in_subnet("subnet")
                .with_provider_name("DFINITY")
            )
            .with_node_entry(
                NodeEntry()
                .with_country("BB")
                .with_status("UP")
                .in_subnet("subnet")
                .with_provider_name("DFINITY")
            )
            .with_node_entry(unhealthy_node)
            .with_extend_nodes(healthy_nodes)
        ).build()

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"
        assert len(output) == 1

        subnet_changes = output[0]

        assert subnet_changes["subnet_id"] == "subnet"

        assert len(subnet_changes["removed"]) == 1
        assert len(subnet_changes["added"]) == 1

        removed = subnet_changes["removed"][0]
        assert removed["node_id"] == unhealthy_node._node_id

        added = subnet_changes["added"][0]

        # Assert that the node that is picked was among the healthy ones.
        assert added["node_id"] in [
            node._node_id
            for node in healthy_nodes
            if node._status in ["UP", "UNASSIGNED"]
        ]

    def test_all_unassigned_nodes_dead(self):
        unhealthy_unassigned_nodes = [
            NodeEntry()
            .with_country("AA")
            .with_status("DOWN")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("BB")
            .with_status("DOWN")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("CC")
            .with_status("DEGRADED")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("DD")
            .with_status("DEGRADED")
            .with_provider_name("DFINITY"),
        ]

        topology = (
            TopologyEntry()
            .with_subnet_type("NNS")
            .with_size(3)
            .with_subnet_id("subnet")
        )

        network_data = (
            NetworkData()
            .with_topology_entry(topology)
            .with_extend_nodes(unhealthy_unassigned_nodes)
            .enforce_health()
        ).build()

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status != "Optimal"

    def test_all_unassigned_dead_but_no_health_constraint_required(self):
        unhealthy_unassigned_nodes = [
            NodeEntry()
            .with_country("AA")
            .with_status("DOWN")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("BB")
            .with_status("DOWN")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("CC")
            .with_status("DEGRADED")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("DD")
            .with_status("DEGRADED")
            .with_provider_name("DFINITY"),
        ]

        topology = (
            TopologyEntry()
            .with_subnet_type("NNS")
            .with_size(3)
            .with_subnet_id("subnet")
        )

        network_data = (
            NetworkData()
            .with_topology_entry(topology)
            .with_extend_nodes(unhealthy_unassigned_nodes)
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data.build())

        assert status == "Optimal"


if __name__ == "__main__":
    unittest.main()
