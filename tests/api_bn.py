import unittest
from tests.test_utils import (
    NetworkData,
    NodeEntry,
    TopologyEntry,
    execute_min_synthetic_nodes_scenario,
)


class TestApiBnConstraints(unittest.TestCase):
    def test_do_not_use_api_bns(self):
        boundary_node = (
            NodeEntry()
            .with_country("BB")
            .with_node_type("API_BOUNDARY")
            .with_provider_name("DFINITY")
        )

        node_bag = [
            NodeEntry()
            .with_country("AA")
            .with_node_type("REPLICA")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("CC")
            .with_node_type("REPLICA")
            .with_provider_name("DFINITY"),
            NodeEntry()
            .with_country("DD")
            .with_node_type("REPLICA")
            .with_provider_name("DFINITY"),
        ]

        topology = TopologyEntry().with_subnet_type("NNS").with_size(3)

        network_data = (
            NetworkData()
            .with_extend_nodes(node_bag)
            .with_node_entry(boundary_node)
            .with_topology_entry(topology)
        ).build()

        output, status = execute_min_synthetic_nodes_scenario(network_data)

        assert status == "Optimal"

        subnet = output[0]

        assert len(subnet["added"]) == 3

        assert boundary_node._node_id not in [
            node["node_id"] for node in subnet["added"]
        ]


if __name__ == "__main__":
    unittest.main()
