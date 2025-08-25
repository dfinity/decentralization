import pandas as pd
import json
from collections import Counter, defaultdict
import logging
from topology_optimizer.utils import get_node_pipeline


def check_filled_subnets(json_data, df_topology):
    errors = []
    for subnet in json_data:
        subnet_id = subnet["subnet_id"]
        expected = df_topology.loc[df_topology["subnet_id"] == subnet_id, "subnet_size"]
        if expected.empty:
            errors.append(f"{subnet_id}: not found in topology")
            continue
        actual = len(subnet["unchanged"]) + len(subnet["added"])
        if actual != expected.values[0]:
            errors.append(f"{subnet_id}: has {actual}, expected {expected.values[0]}")
    return errors


def check_unique_node_usage(json_data):
    node_ids = [
        node["node_id"]
        for subnet in json_data
        for group in ["unchanged", "added"]
        for node in subnet.get(group, [])
    ]
    return [nid for nid, count in Counter(node_ids).items() if count > 1]


def check_subnet_limits(json_data, df_topology, node_metadata):
    violations = defaultdict(list)
    dimensions = [
        ("node_provider", "subnet_limit_node_provider"),
        ("data_center", "subnet_limit_data_center"),
        ("data_center_provider", "subnet_limit_data_center_provider"),
        ("country", "subnet_limit_country"),
    ]

    nns_exempt_values = {
        "node_provider": {"DFINITY"},
        "data_center": {"sh1", "zh2"},
        "data_center_provider": {"Everyware", "Digital Realty"},
    }

    for subnet in json_data:
        subnet_id = subnet["subnet_id"]
        subnet_type = subnet.get("subnet_type")
        row = df_topology[df_topology["subnet_id"] == subnet_id]
        if row.empty:
            violations["missing_topology"].append(subnet_id)
            continue
        limits = row.iloc[0]

        nodes = [
            node.copy()
            for group in ["unchanged", "added"]
            for node in subnet.get(group, [])
        ]
        for node in nodes:
            node_id = node["node_id"]
            meta = node_metadata.get(node_id, {})
            node.update(meta)

        for field, limit_col in dimensions:
            limit = limits.get(limit_col)
            if pd.isna(limit):
                continue
            values = [
                node.get(field)
                for node in nodes
                if node.get(field) is not None
                and not (
                    subnet_type == "NNS"
                    and field in nns_exempt_values
                    and node.get(field) in nns_exempt_values[field]
                )
            ]
            counts = Counter(values)
            for val, count in counts.items():
                if count > limit:
                    violations[f"{field}_limit"].append(
                        f"{subnet_id}: {field}={val} used {count}x (limit={limit})"
                    )
    return dict(violations)


def check_european_subnet_countries(json_data, node_metadata):
    EU_COUNTRIES = {
        "AT",
        "BE",
        "BG",
        "CH",
        "CY",
        "CZ",
        "DE",
        "DK",
        "EE",
        "ES",
        "FI",
        "FR",
        "GR",
        "HR",
        "HU",
        "IE",
        "IS",
        "IT",
        "LI",
        "LT",
        "LU",
        "LV",
        "MT",
        "NL",
        "NO",
        "PL",
        "PT",
        "RO",
        "SE",
        "SI",
        "SK",
        "UK",
        "GB",
        "IM",
    }

    violations = []
    for subnet in json_data:
        if subnet.get("subnet_type") != "European Subnet":
            continue
        subnet_id = subnet["subnet_id"]
        nodes = [
            node.copy()
            for group in ["unchanged", "added"]
            for node in subnet.get(group, [])
        ]
        for node in nodes:
            node_id = node["node_id"]
            country = node_metadata.get(node_id, {}).get("country")
            if not country:
                violations.append(f"{subnet_id}: node {node_id} missing country info")
            elif country not in EU_COUNTRIES:
                violations.append(
                    f"{subnet_id}: node {node_id} assigned to non-EU country '{country}'"
                )
    return violations


def check_nns_special_rules(json_data, node_metadata):
    violations = []
    for subnet in json_data:
        if subnet.get("subnet_type") != "NNS":
            continue

        subnet_id = subnet["subnet_id"]
        nodes = [
            node.copy()
            for group in ["unchanged", "added"]
            for node in subnet.get(group, [])
        ]
        for node in nodes:
            node_id = node["node_id"]
            meta = node_metadata.get(node_id, {})
            node.update(meta)

        provider_counts = Counter([n.get("node_provider") for n in nodes])
        dfinity_count = provider_counts.get("DFINITY", 0)
        if dfinity_count != 3:
            violations.append(
                f"{subnet_id}: DFINITY must appear exactly 3x (found {dfinity_count})"
            )

        dc_counts = Counter([n.get("data_center") for n in nodes])
        for dc in ["sh1", "zh2"]:
            if dc_counts.get(dc, 0) > 2:
                violations.append(
                    f"{subnet_id}: data_center {dc} used {dc_counts[dc]}x (max 2)"
                )

        dcp_counts = Counter([n.get("data_center_provider") for n in nodes])
        for dcp in ["Everyware", "Digital Realty"]:
            if dcp_counts.get(dcp, 0) > 2:
                violations.append(
                    f"{subnet_id}: data_center_provider {dcp} used {dcp_counts[dcp]}x (max 2)"
                )

    return violations


def validate_subnet_assignment(
    node_file: str, topology_file: str, json_file: str, pipeline_file: str
) -> dict:
    df_nodes = pd.read_csv(node_file)
    pipeline = get_node_pipeline(pipeline_file)

    df_nodes["data_center"] = df_nodes["dc_id"]
    df_nodes["data_center_provider"] = df_nodes["owner"]
    df_nodes["country"] = df_nodes["region"].apply(
        lambda r: r.split(",")[1] if isinstance(r, str) and "," in r else None
    )

    # Apply after postprocessing because of transformations above
    df_nodes = pd.concat([df_nodes, pipeline], ignore_index=True)
    node_metadata = df_nodes.set_index("node_id")[
        ["data_center", "data_center_provider", "country"]
    ].to_dict(orient="index")

    df_topology = pd.read_csv(topology_file)
    df_topology["is_sev"] = df_topology["is_sev"].map({"TRUE": True, "FALSE": False})

    with open(json_file, "r") as f:
        json_data = json.load(f)

    violations = {
        "subnet_size_mismatches": check_filled_subnets(json_data, df_topology),
        "duplicate_nodes": check_unique_node_usage(json_data),
        "limit_violations": check_subnet_limits(json_data, df_topology, node_metadata),
        "european_subnet_violations": check_european_subnet_countries(
            json_data, node_metadata
        ),
        "nns_special_violations": check_nns_special_rules(json_data, node_metadata),
    }

    if any(v for v in violations.values()):
        raise ValueError(
            f"Subnet assignment validation failed for {json_file}:\n"
            f"{json.dumps(violations, indent=2)}"
        )

    logging.info(f"✅ Subnet assignment validation passed for {json_file}")
    return violations
