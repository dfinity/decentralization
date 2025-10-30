import unittest

from tests.test_utils import (
    NetworkData,
    NodeEntry,
    TopologyEntry,
    execute_min_synthetic_nodes_scenario,
)


class SpareCapacityTestScenarios(unittest.TestCase):
    def test_rebalance_if_possible(self):
        topology = (
            TopologyEntry()
            .with_subnet_type("NNS")
            .with_subnet_id("subnet")
            .with_size(10)
            .with_country_limit(10)
            .with_dc_limit(10)
            .with_dc_provider_limit(10)
            .with_node_provider_limit(10)
        )

        np_1 = []
        for _ in range(10):
            np_1.append(
                NodeEntry()
                .with_provider_name("DFINITY")
                .with_country("AA")
                .with_data_center("aa1")
                .with_owner("owner")
                .in_subnet("subnet")
            )

        np_2 = []
        for _ in range(5):
            np_2.append(
                NodeEntry()
                .with_provider_name("Other np")
                .with_country("AA")
                .with_owner("owner")
                .with_data_center("aa1")
            )

        # Initially don't enable the spare capacity feature
        network_data = (
            NetworkData()
            .with_extend_nodes(np_1)
            .with_extend_nodes(np_2)
            .with_topology_entry(topology)
            # Needed to evade the default special limit of exactly 3 dfinity nodes in NNS
            .with_special_limit("subnet", "node_provider", "DFINITY", 1000, "lt")
        )

        output, status = execute_min_synthetic_nodes_scenario(network_data.build())

        assert status == "Optimal"
        # There should be no swaps because the current topology is
        # optimal
        for changes in output:
            assert len(changes["added"]) == 0
            assert len(changes["removed"]) == 0

        # Now enable the feature and try again
        network_data = network_data.with_spare_node_ratio(0.1)

        output, status = execute_min_synthetic_nodes_scenario(network_data.build())
        assert status == "Optimal"

        only_swap = output[0]
        added = only_swap["added"]
        removed = only_swap["removed"]

        assert len(added) == 1
        assert len(removed) == 1

        assert added[0]["node_provider"] == "Other np"
        assert removed[0]["node_provider"] == "DFINITY"

    def test_failure_if_ratio_outside_bounds(self):
        topology = (
            TopologyEntry()
            .with_subnet_type("NNS")
            .with_subnet_id("subnet")
            .with_size(3)
        )

        nodes = [
            NodeEntry()
            .with_provider_name("DFINITY")
            .in_subnet("subnet")
            .with_country("aa")
            for _ in range(3)
        ]

        network_data = (
            NetworkData().with_extend_nodes(nodes).with_topology_entry(topology)
        )

        ratios_to_test = [-15, 15, -0.4, 1.2]

        for ratio in ratios_to_test:
            network_data = network_data.with_spare_node_ratio(ratio)

            expected_exception = None
            try:
                _, _ = execute_min_synthetic_nodes_scenario(network_data.build())
            except ValueError as e:
                expected_exception = e

            assert expected_exception is not None
            assert "Spare node ratio has to be a float between 0 and 1" in str(
                expected_exception
            )


if __name__ == "__main__":
    unittest.main()
