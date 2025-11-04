from typing import Any, Dict, List

import pandas as pd

from topology_optimizer.utils import (
    create_node_dataframe,
    generate_synthetic_countries,
    generate_synthetic_nodes,
    get_existing_assignment,
    mark_blacklisted_nodes,
    post_process_node_providers,
)


def prepare_data(
    df_nodes: pd.DataFrame,
    df_node_pipeline: pd.DataFrame,
    network_topology: pd.DataFrame,
    blacklist: Dict[str, set],
    no_synthetic_countries: int,
    enforce_sev_constraint: bool,
    enforce_health_constraint: bool,
    enforce_blacklist_constraint: bool,
    cluster_scenario: Dict[str, List[str]],
    cluster_scenario_name: str,
    enforce_per_node_provider_assignation: bool,
    sev_node_providers: List[str],
    special_limits: dict[int, dict[str, dict[str, (int, str)]]] = None,
) -> Dict[str, Any]:
    # Remove everything that is not a replica
    df_nodes = df_nodes[df_nodes["node_type"] != "API_BOUNDARY"]
    # Preprocessing
    current_nodes = create_node_dataframe(df_nodes, sev_node_providers)

    # Apply standard corrections
    current_nodes.loc[
        current_nodes["node_provider"].str.startswith("DFINITY"), "node_provider"
    ] = "DFINITY"

    # Combine with pipeline
    current_nodes = pd.concat([current_nodes, df_node_pipeline], ignore_index=True)

    # Add synthetic nodes
    current_countries = list(current_nodes["country"].unique())
    synthetic_countries = generate_synthetic_countries(no_synthetic_countries)
    synthetic_nodes_df = generate_synthetic_nodes(synthetic_countries)
    node_all = pd.concat([current_nodes, synthetic_nodes_df], ignore_index=True)

    # Apply clustering
    node_all = post_process_node_providers(node_all, cluster_scenario)

    node_all = mark_blacklisted_nodes(node_all, blacklist, cluster_scenario_name)

    # Indices and metadata
    node_all_indices = node_all.index.tolist()
    synthetic_node_indicator = node_all["is_synthetic"].tolist()
    subnet_indices = network_topology.index.tolist()

    unique_node_providers = node_all["node_provider"].unique().tolist()
    unique_preclustering_node_providers = (
        node_all["original_node_provider"].unique().tolist()
    )
    node_providers_indices = list(range(len(unique_node_providers)))

    all_countries = current_countries + synthetic_countries
    all_countries_indices = list(range(len(all_countries)))

    data_center_all = list(node_all["data_center"].unique())
    data_center_all_indices = list(range(len(data_center_all)))

    data_center_provider_all = list(node_all["data_center_provider"].unique())
    data_center_provider_all_indices = list(range(len(data_center_provider_all)))

    current_assignment = get_existing_assignment(df_nodes, network_topology)

    if special_limits is None:
        special_limits = default_special_limits(network_topology)

    return {
        "network_topology": network_topology,
        "enforce_sev_constraint": enforce_sev_constraint,
        "enforce_health_constraint": enforce_health_constraint,
        "enforce_blacklist_constraint": enforce_blacklist_constraint,
        "subnet_indices": subnet_indices,
        "node_df": node_all,
        "node_indices": node_all_indices,
        "node_df_current": current_nodes,
        "node_df_synthetic": synthetic_nodes_df,
        "synthetic_node_indicator": synthetic_node_indicator,
        "node_provider_list": unique_node_providers,
        "node_provider_list_before_clustering": unique_preclustering_node_providers,
        "node_provider_indices": node_providers_indices,
        "country_list": all_countries,
        "country_indices": all_countries_indices,
        "data_center_list": data_center_all,
        "data_center_indices": data_center_all_indices,
        "data_center_provider_list": data_center_provider_all,
        "data_center_provider_indices": data_center_provider_all_indices,
        "current_assignment": current_assignment,
        "enforce_per_node_provider_assignation": enforce_per_node_provider_assignation,
        "special_limits": special_limits,
    }


def default_special_limits(
    network_topology: pd.DataFrame,
) -> dict[str, dict[str, dict[str, (int, str)]]]:
    nns = network_topology.loc[
        network_topology["subnet_type"] == "NNS", "subnet_id"
    ].iloc[0]
    return {
        nns: {
            "node_provider": {"DFINITY": (3, "eq")},
            "data_center": {"sh1": (2, "lt"), "zh2": (2, "lt")},
            "data_center_provider": {
                "Everyware": (2, "lt"),
                "Digital Realty": (2, "lt"),
            },
        },
        "default": {"node_provider": {"DFINITY": (1, "eq")}},
    }
