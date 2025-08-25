from typing import List, Dict, Self, Any
import uuid
from topology_optimizer.utils import ALLOWED_FEATURES, parse_solver_result
from topology_optimizer.data_preparation import prepare_data
from topology_optimizer.linear_solver import (
    solver_model_minimize_swaps,
    ATTRIBUTE_NAMES,
)
import pandas as pd


class TopologyEntry:
    _subnet_type: str
    _subnet_id: str
    _number_of_subnets: int
    _subnet_size: int
    _is_sev: bool
    _subnet_limit_node_provider: int
    _subnet_limit_data_center: int
    _subnet_limit_data_center_provider: int
    _subnet_limit_country: int

    def __init__(self):
        self._is_sev = False
        self._subnet_limit_node_provider = 1
        self._subnet_limit_data_center = 1
        self._subnet_limit_data_center_provider = 1
        self._subnet_limit_country = 2
        self._subnet_size = 13
        self._subnet_id = uuid.uuid4()

    def with_subnet_type(self, type: str) -> Self:
        self._subnet_type = type
        return self

    def with_subnet_id(self, id: str) -> Self:
        self._subnet_id = id
        return self

    def is_sev(self) -> Self:
        self._is_sev = True
        return self

    def with_node_provider_limit(self, limit: int) -> Self:
        if limit < 1:
            raise ValueError("Invalid limit for node provider:", limit)

        self._subnet_limit_node_provider = limit
        return self

    def with_dc_limit(self, limit: int) -> Self:
        if limit < 1:
            raise ValueError("Invalid limit for dc:", limit)

        self._subnet_limit_data_center = limit
        return self

    def with_dc_provider_limit(self, limit: int) -> Self:
        if limit < 1:
            raise ValueError("Invalid limit for dc provider:", limit)

        self._subnet_limit_data_center_provider = limit
        return self

    def with_country_limit(self, limit: int) -> Self:
        if limit < 1:
            raise ValueError("Invalid limit for country:", limit)

        self._subnet_limit_country = limit
        return self

    def with_size(self, size: int) -> Self:
        if size < 1:
            raise ValueError("Invalid size for a subnet:", size)

        self._subnet_size = size
        return self


class PipelineEntry:
    _node_id: str
    _node_provider: str
    _data_center: str
    _data_center_provider: str
    _country: str
    _is_sev: bool

    def __init__(self):
        self._is_sev = False
        self._node_id = str(uuid.uuid4())

    def with_provider(self, provider: str) -> Self:
        self._node_provider = provider
        return self

    def with_dc_provider(self, provider: str) -> Self:
        self._data_center_provider = provider
        return self

    def with_country(self, country: str) -> Self:
        self._country = country
        return self

    def is_sev(self) -> Self:
        self._is_sev = True
        return self

    def with_id(self, id: str) -> Self:
        self._node_id = id
        return self


ALLOWED_TYPES = ["REPLICA", "API_BOUNDARY"]
ALLOWED_STATUSES = ["UP", "DOWN", "DEGRADED", "UNASSIGNED"]


class NodeEntry:
    _node_id: str
    _dc_id: str
    _node_operator_id: str
    _node_provider_id: str
    _node_provider_name: str
    _node_type: str
    _owner: str
    _country: str
    _status: str
    _subnet_id: str

    def __init__(self):
        self._node_id = str(uuid.uuid4())
        self._node_operator_id = self._dc_id = str(uuid.uuid4())
        self._node_provider_id = self._node_provider_name = self._owner = str(
            uuid.uuid4()
        )
        self._node_type = "REPLICA"
        self._status = "UP"
        self._subnet_id = pd.NA

    def with_operator(self, operator: str) -> Self:
        self._node_operator_id = operator
        return self

    def with_data_center(self, data_center: str) -> Self:
        self._dc_id = data_center
        return self

    def with_provider_name(self, provider_name: str) -> Self:
        self._node_provider_name = provider_name
        self._node_provider_id = provider_name
        return self

    def with_node_type(self, type: str) -> Self:
        if type not in ALLOWED_TYPES:
            raise ValueError(
                "Invalid node type '", type, "'. Allowed node types:", ALLOWED_TYPES
            )

        self._node_type = type
        return self

    def with_owner(self, owner: str) -> Self:
        self._owner = owner
        return self

    def with_country(self, country_code: str) -> Self:
        self._region = "," + country_code + ","
        return self

    def with_status(self, status: str) -> Self:
        if status not in ALLOWED_STATUSES:
            raise ValueError(
                "Invalid status '",
                status,
                "'. Allowed node statuses:",
                ALLOWED_STATUSES,
            )

        self._status = status
        return self

    def in_subnet(self, subnet: str) -> Self:
        self._subnet_id = subnet
        return self


class NetworkData:
    _nodes: List[NodeEntry]
    _pipeline: List[PipelineEntry]
    _network_topology: List[TopologyEntry]
    _blacklist: Dict[str, set]
    _number_synthetic_countries: int
    _sev_node_providers: List[str]

    _enforce_sev_constraint: bool
    _enforce_health_constraint: bool
    _enforce_blacklist_constraint: bool
    _enforce_per_node_provider_assignation: bool

    _cluster_scenario: Dict[str, List[str]]
    _cluster_scenario_name: str
    _special_limits: Dict[int, dict[str, dict[str, (int, str)]]]

    def __init__(self):
        self._cluster_scenario_name = str(uuid.uuid4())
        self._enforce_sev_constraint = False
        self._enforce_blacklist_constraint = False
        self._enforce_per_node_provider_assignation = False
        self._enforce_health_constraint = False

        self._nodes = list()
        self._pipeline = list()
        self._network_topology = list()
        self._number_synthetic_countries = 0
        self._sev_node_providers = list()
        self._cluster_scenario = dict()
        self._special_limits = dict()

        self._blacklist = dict()
        for feature in ALLOWED_FEATURES:
            self._blacklist[ALLOWED_FEATURES[feature]] = set()

    def enforce_sev(self) -> Self:
        self._enforce_sev_constraint = True
        return self

    def enforce_health(self) -> Self:
        self._enforce_health_constraint = True
        return self

    def enforce_blacklist(self) -> Self:
        self._enforce_blacklist_constraint = True
        return self

    def enforce_per_node_provider_assignation(self) -> Self:
        self._enforce_per_node_provider_assignation = True
        return self

    def with_synthetic_countries(self, num_countries: int) -> Self:
        self._number_synthetic_countries = num_countries
        return self

    def with_topology_entry(self, entry: TopologyEntry) -> Self:
        self._network_topology.append(entry)
        return self

    def with_extend_topology_entries(self, entries: List[TopologyEntry]) -> Self:
        for entry in entries:
            self.with_topology_entry(entry)

        return self

    def with_cluster(self, cluster_name: str, clustered_providers: List[str]) -> Self:
        if cluster_name in self._cluster_scenario:
            raise ValueError("Cluster with name", cluster_name, "was already defined.")

        self._cluster_scenario[cluster_name] = clustered_providers
        return self

    def with_extend_clusters(self, entries: Dict[str, List[str]]) -> Self:
        for key in entries:
            self.with_cluster(key, entries[key])

        return self

    def with_blacklist_entry(self, feature: str, value: str) -> Self:
        if feature not in ALLOWED_FEATURES:
            raise ValueError(
                "Feature '",
                feature,
                "' is not allowed. Allowed features are:",
                ALLOWED_FEATURES,
            )

        self._blacklist[ALLOWED_FEATURES[feature]].add(value)
        return self

    def with_node_entry(self, node: NodeEntry) -> Self:
        self._nodes.append(node)
        return self

    def with_extend_nodes(self, nodes: list[NodeEntry]) -> Self:
        for node in nodes:
            self.with_node_entry(node)
        return self

    def with_pipeline_entry(self, node: PipelineEntry) -> Self:
        self._pipeline.append(node)
        return self

    def with_extend_pipeline(self, nodes: List[PipelineEntry]) -> Self:
        for node in nodes:
            self.with_pipeline_entry(node)

        return self

    def with_sev_provider(self, provider: str) -> Self:
        self._sev_node_providers.append(provider)
        return self

    def with_extend_sev_providers(self, providers: list[str]) -> Self:
        for provider in providers:
            self.with_sev_provider(provider)

        return self

    def with_special_limit(
        self, subnet: str, attr: str, key: str, value: int, operator: str
    ) -> Self:
        subnet_index = None
        for index, topology_subnet in enumerate(self._network_topology):
            if topology_subnet._subnet_id == subnet:
                subnet_index = index
                break

        if subnet_index is None:
            raise ValueError(f"Subnet {subnet} not found")

        if attr not in ATTRIBUTE_NAMES:
            raise ValueError(f"Attribute {attr} is unknown")

        if subnet_index not in self._special_limits:
            self._special_limits[subnet_index] = {}

        if attr not in self._special_limits[subnet_index]:
            self._special_limits[subnet_index][attr] = {}

        self._special_limits[subnet_index][attr][key] = (value, operator)

        return self

    def _transform(self, records: list[Any]) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [{k.lstrip("_"): v for k, v in vars(n).items()} for n in records]
        )

    def build(self) -> dict[str, Any]:
        return prepare_data(
            df_nodes=self._transform(self._nodes),
            df_node_pipeline=self._transform(self._pipeline),
            network_topology=self._transform(self._network_topology),
            blacklist=self._blacklist,
            no_synthetic_countries=self._number_synthetic_countries,
            enforce_sev_constraint=self._enforce_sev_constraint,
            enforce_health_constraint=self._enforce_health_constraint,
            enforce_blacklist_constraint=self._enforce_blacklist_constraint,
            enforce_per_node_provider_assignation=self._enforce_per_node_provider_assignation,
            cluster_scenario=self._cluster_scenario,
            cluster_scenario_name=self._cluster_scenario_name,
            sev_node_providers=self._sev_node_providers,
            special_limits=self._special_limits
            if len(self._special_limits) > 0
            else None,
        )


def execute_min_synthetic_nodes_scenario(network_data: Any) -> tuple[Any, str]:
    result, status = solver_model_minimize_swaps(network_data)

    output = parse_solver_result(network_data, result)

    # Print the output of the solve here. This will get captured by the test executor
    # and will be displayed only if the test fails.
    print("Output of the solver:\n", result["solver_output"])

    return (output, status)
