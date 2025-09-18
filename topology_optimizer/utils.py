#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
from typing import Dict, List, Optional

import pandas as pd
import yaml

# Constants
UNAVAILABLE_STATUSES = {"DOWN", "DEGRADED"}

ALLOWED_FEATURES = {
    "node_id": "node_id",
    "node_provider": "node_provider_id",
    "node_operator": "node_operator",
    "data_center": "data_center",
    "data_center_provider": "data_center_provider",
}


###############################################################################
# Helper functions


def generate_synthetic_countries(no_synthetic_countries: int) -> List[str]:
    """
    Generate a list of synthetic country identifiers.
    """
    return [f"C{i + 1}" for i in range(no_synthetic_countries)]


def post_process_node_providers(
    df: pd.DataFrame, provider_groups: Dict[str, List[str]]
) -> pd.DataFrame:
    """
    Consolidate node providers into groups as defined in the provider_groups mapping.
    """
    df_processed = df.copy()
    # Needed for the per node feature assignation because the actual node provider data is lost here.
    df_processed["original_node_provider"] = df_processed["node_provider"]
    for new_provider, original_names in provider_groups.items():
        mask = df_processed["node_provider"].isin(original_names)
        df_processed.loc[mask, "node_provider"] = new_provider
    return df_processed


def create_node_dataframe(
    df_nodes: pd.DataFrame, sev_node_providers: List[str]
) -> pd.DataFrame:
    """
    Convert raw node info into structured format with availability and SEV flag.
    """
    new_records = []

    for idx, row in df_nodes.iterrows():
        region_parts = str(row["region"]).split(",")
        region = region_parts[1].strip() if len(region_parts) > 1 else "UNKNOWN"

        new_records.append(
            {
                "node_id": row["node_id"],
                "node_provider": row["node_provider_name"],
                "data_center": row["dc_id"],
                "data_center_provider": row["owner"],
                "node_operator": row["node_operator_id"],
                "node_provider_id": row["node_provider_id"],
                "country": region,
                "is_synthetic": False,
                "is_sev": row["node_provider_name"] in sev_node_providers,
                "node_type": row["node_type"],
                "is_available": str(row["status"]).strip().upper()
                not in UNAVAILABLE_STATUSES,
            }
        )

    return pd.DataFrame(new_records)


def generate_synthetic_nodes(
    synthetic_countries: List[str],
    no_node_provider_per_country: int = 5,
    no_nodes_per_provider: int = 4,
) -> pd.DataFrame:
    """
    Generate a DataFrame of synthetic nodes based on synthetic countries.
    """
    total_providers = no_node_provider_per_country * len(synthetic_countries)
    synthetic_node_provider = [f"SYNTHETIC_NP{i + 1}" for i in range(total_providers)]
    synthetic_dc = [f"SYNTHETIC_DC{i + 1}" for i in range(total_providers)]
    synthetic_dc_provider = [f"SYNTHETIC_DCP{i + 1}" for i in range(total_providers)]

    data = []
    provider_index = 0

    for country in synthetic_countries:
        for _ in range(no_node_provider_per_country):
            for node_idx in range(no_nodes_per_provider):
                node_id = f"synthetic-{country}-{provider_index}-{node_idx}"
                data.append(
                    {
                        "node_id": node_id,
                        "node_provider": synthetic_node_provider[provider_index],
                        "data_center": synthetic_dc[provider_index],
                        "data_center_provider": synthetic_dc_provider[provider_index],
                        "node_operator": None,
                        "node_provider_id": None,
                        "country": country,
                        "is_synthetic": True,
                        "is_sev": True,
                        "node_type": "REPLICA",
                        "is_available": True,
                    }
                )
            provider_index += 1

    return pd.DataFrame(data)


def calculate_nakamoto_for_attribute(
    df_attribute_allocations: pd.DataFrame,
) -> Dict[int, int]:
    """
    Compute Nakamoto coefficient for a given attribute allocation DataFrame.
    """
    coefficients = {}
    for subnet_idx in df_attribute_allocations["subnet_index"].unique():
        entries = df_attribute_allocations[
            df_attribute_allocations["subnet_index"] == subnet_idx
        ]
        if not entries.empty:
            total = entries["allocation_value"].sum()
            sorted_entries = entries.sort_values(by="allocation_value", ascending=False)
            cumulative = 0
            nakamoto = 0
            for _, row in sorted_entries.iterrows():
                cumulative += row["allocation_value"]
                nakamoto += 1
                if cumulative > total / 3:
                    break
            coefficients[subnet_idx] = nakamoto
        else:
            coefficients[subnet_idx] = 0
    return coefficients


def calculate_nakamoto_coefficient(
    result: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[int, int]]:
    """
    Compute Nakamoto coefficients for all attributes.
    """
    attributes = ["node_provider", "data_center", "data_center_provider", "country"]
    return {
        f"nakamoto_coefficients_{attr}": calculate_nakamoto_for_attribute(
            result[f"df_{attr}_allocations"]
        )
        for attr in attributes
    }


def get_target_topology(file_path: str) -> pd.DataFrame:
    """
    Expand and process a target topology file to include Nakamoto metrics.
    """
    df = pd.read_csv(file_path)
    df["is_sev"] = df["is_sev"].astype(bool)
    df["number_of_subnets"] = df["number_of_subnets"].astype(int)
    df["subnet_size"] = df["subnet_size"].astype(int)

    df_expanded = (
        df.loc[df.index.repeat(df["number_of_subnets"])].copy().reset_index(drop=True)
    )
    df_expanded.drop(columns="number_of_subnets", inplace=True)

    attributes = ["node_provider", "data_center", "data_center_provider", "country"]
    for attr in attributes:
        limit_col = f"subnet_limit_{attr}"
        target_col = f"nakamoto_target_{attr}"
        df_expanded[limit_col] = df_expanded[limit_col].astype(int)
        df_expanded[target_col] = (
            df_expanded["subnet_size"] // (3 * df_expanded[limit_col])
        ) + 1

    return df_expanded


def get_subnet_limit(
    network_topology: pd.DataFrame, subnet_index: int, attribute: str
) -> int:
    return network_topology.at[subnet_index, f"subnet_limit_{attribute}"]


def get_node_pipeline(file_path: str) -> pd.DataFrame:
    """
    Load pipeline node configurations from a CSV.
    Each row in the CSV corresponds to a single node and must include a unique `node_id`.
    """

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        logging.warning("Pipeline CSV is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    except FileNotFoundError:
        logging.warning(
            f"Pipeline CSV not found at {file_path}. Returning empty DataFrame."
        )
        return pd.DataFrame()

    required_columns = {
        "node_id",
        "node_provider",
        "data_center",
        "data_center_provider",
        "country",
        "is_sev",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in input file: {missing}")

    # Enrich the DataFrame with default fields
    df["node_operator"] = None
    df["node_provider_id"] = None
    df["is_synthetic"] = False
    df["is_available"] = True
    df["node_type"] = "REPLICA"

    return df


def get_existing_assignment(
    df_nodes: pd.DataFrame, network_topology: pd.DataFrame
) -> List[Optional[int]]:
    """
    Map node assignments from subnet IDs to positional indices in the network topology.
    """
    nt_pos = network_topology.reset_index(drop=True)
    subnet_to_pos = pd.Series(nt_pos.index, index=nt_pos["subnet_id"]).to_dict()

    assignment = []
    for node_pos, subnet_id in enumerate(df_nodes["subnet_id"].values):
        if pd.isna(subnet_id):
            assignment.append(None)
        else:
            try:
                assignment.append(subnet_to_pos[subnet_id])
            except KeyError:
                raise KeyError(
                    f"Subnet_id {subnet_id!r} (node row {node_pos}) is not in network_topology['subnet_id']."
                )
    return assignment


def parse_solver_result(network_data, solver_result):
    node_df_all = network_data["node_df"]

    current_assignment = network_data["current_assignment"]
    topology = network_data["network_topology"]

    final_df = solver_result["df_node_allocations"][["node_index", "subnet_index"]]
    final_assignment = dict(zip(final_df["node_index"], final_df["subnet_index"]))

    dropped = set(solver_result["deviation_nodes"]["dropped"])
    moved = set(solver_result["deviation_nodes"]["moved"])
    new_assign = set(solver_result["deviation_nodes"]["new_assign"])

    before = {}
    after = {}

    for i, s in enumerate(current_assignment):
        if s is not None:
            before.setdefault(s, set()).add(i)
    for i, s in final_assignment.items():
        after.setdefault(s, set()).add(i)

    output = []
    for subnet_idx in topology.index:
        subnet_id = topology.loc[subnet_idx, "subnet_id"]
        subnet_type = topology.loc[subnet_idx, "subnet_type"]

        b = before.get(subnet_idx, set())
        a = after.get(subnet_idx, set())

        unchanged = b & a
        removed = b - a
        added = a - b

        def node_info(i, change=None):
            row = node_df_all.loc[i]
            return {
                "node_id": row["node_id"],
                "node_provider": row["node_provider"],
                "change_type": change,
            }

        unchanged_nodes = [node_info(i) for i in sorted(unchanged)]

        removed_nodes = [
            node_info(i, "dropped" if i in dropped else "moved_away")
            for i in sorted(removed)
            if i in dropped or i in moved
        ]

        added_nodes = [
            node_info(i, "newly_assigned" if i in new_assign else "moved_in")
            for i in sorted(added)
            if i in new_assign or i in moved
        ]

        output.append(
            {
                "subnet_id": subnet_id,
                "subnet_type": subnet_type,
                "unchanged": unchanged_nodes,
                "removed": removed_nodes,
                "added": added_nodes,
            }
        )

    return output


def export_subnet_node_changes_to_json(
    network_data, solver_result, output_path="./output/subnet_node_changes.json"
):
    """
    Export subnet changes as JSON including unchanged, removed, and added nodes.
    """
    output = parse_solver_result(network_data, solver_result)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Exported subnet changes to: {output_path}")


def load_blacklist(file_path: str) -> Dict[str, set]:
    """
    Load a blacklist YAML file and return a dictionary mapping feature → set of blacklisted values.

    Supports features: 'node_id', 'node_provider', 'node_operator', 'data_center', 'data_center_provider'
    """

    # Map of values from file to values in dataframes.
    blacklist = {feature: set() for feature in ALLOWED_FEATURES.values()}

    try:
        with open(file_path, "r") as f:
            data = yaml.safe_load(f) or {}  # <-- prevent data from being None

        features = data.get("features") or []  # <-- ensure it's an iterable
        for entry in features:
            feature = entry.get("feature")
            value = entry.get("value")
            if feature in ALLOWED_FEATURES and value is not None:
                blacklist[ALLOWED_FEATURES[feature]].add(value)

    except Exception as e:
        logging.warning(f"Failed to load blacklist from {file_path}: {e}")

    return blacklist


def mark_blacklisted_nodes(
    df: pd.DataFrame, blacklist: dict, scenario_name: str
) -> pd.DataFrame:
    """
    Marks nodes in the DataFrame with a boolean flag 'is_blacklisted' if they match any blacklist criteria.
    Also logs the blacklisted nodes and reasons to ./output/blacklisted_nodes.csv.
    """
    df = df.copy()
    df["is_blacklisted"] = False
    reasons = [[] for _ in range(len(df))]

    print("Blacklist:\n", blacklist)

    for feature, values in blacklist.items():
        if feature not in df.columns:
            print(
                "Skipping blacklisting of a feature, because it's not present:", feature
            )
            continue
        for value in values:
            mask = df[feature] == value
            df.loc[mask, "is_blacklisted"] = True
            for i in df[mask].index:
                reasons[i].append(f"{feature} == {value}")

    df["blacklist_reason"] = ["; ".join(r) if r else "" for r in reasons]

    num_blacklisted = df["is_blacklisted"].sum()
    logging.info(f"{num_blacklisted} nodes marked as blacklisted out of {len(df)}")

    if num_blacklisted > 0:
        os.makedirs("./output", exist_ok=True)
        filename = f"./output/blacklisted_nodes_{scenario_name}.csv"
        log_columns = [
            "node_id",
            "node_provider",
            "data_center",
            "data_center_provider",
            "blacklist_reason",
        ]
        out_df = df[df["is_blacklisted"]][log_columns].drop_duplicates()
        out_df.to_csv(filename, index=False)
        logging.info(f"Blacklisted node details written to {filename}")

    return df
