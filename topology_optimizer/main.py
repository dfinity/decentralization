#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network Topology Analysis and Visualization Script.
"""

import sys
import os
import traceback

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
from enum import Enum
from pathlib import Path
from topology_optimizer.testing import validate_subnet_assignment
import click
import pandas as pd
from topology_optimizer.data_preparation import prepare_data
from topology_optimizer.linear_solver import (
    solver_model_minimize_nodes,
    solver_model_minimize_swaps,
)
from topology_optimizer.visualization import (
    visualize_input_data,
    visualize_model_output,
    visualize_current_node_allocation,
    visualize_subnet_changes_from_json,
    visualize_provider_utilization,
)
from topology_optimizer.utils import (
    get_target_topology,
    get_node_pipeline,
    export_subnet_node_changes_to_json,
    load_blacklist,
)
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Set working directory to the project root (parent of this script's folder)
# This ensures that all relative paths resolve from the project root, not the script folder
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, ".."))
os.chdir(project_root)


class Mode(str, Enum):
    MIN_SYNTHETIC = "minimize_new_nodes"
    MIN_SWAPS = "minimize_node_swaps"


def load_and_prepare_data(
    file_path_current_nodes: Path,
    cluster_file: Path,
    topology_file: Path,
    blacklist_file: Path,
    node_pipeline_file: Path,
    no_synthetic_countries: int,
    enforce_sev_constraint: bool,
    enforce_health_constraint: bool,
    enforce_blacklist_constraint: bool,
    enforce_per_node_provider_assignation: bool,
    special_limits_file: Optional[Path],
) -> pd.DataFrame:
    df_nodes = pd.read_csv(file_path_current_nodes)
    df_node_pipeline = get_node_pipeline(node_pipeline_file)
    network_topology = get_target_topology(topology_file)
    blacklist = load_blacklist(blacklist_file)
    try:
        with open(cluster_file, "r") as f:
            cluster_scenario = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise ValueError(f"Failed to load cluster scenario from {cluster_file}: {e}")

    script_dir = Path(__file__).resolve().parent
    sev_csv_path = script_dir.parent / "data" / "sev_providers.csv"

    try:
        df = pd.read_csv(sev_csv_path)
        sev_providers = df["provider_name"].tolist()
    except Exception as e:
        logging.warning(f"Failed to load SEV providers from {sev_csv_path}: {e}")
        sev_providers = []

    special_limits = None
    if special_limits_file is not None:
        with open(special_limits_file, "r") as f:
            special_limits = json.load(f)

    return prepare_data(
        df_nodes,
        df_node_pipeline,
        network_topology,
        blacklist=blacklist,
        no_synthetic_countries=no_synthetic_countries,
        enforce_sev_constraint=enforce_sev_constraint,
        enforce_health_constraint=enforce_health_constraint,
        enforce_blacklist_constraint=enforce_blacklist_constraint,
        cluster_scenario=cluster_scenario,
        cluster_scenario_name=Path(cluster_file).stem,
        enforce_per_node_provider_assignation=enforce_per_node_provider_assignation,
        sev_node_providers=sev_providers,
        special_limits=special_limits,
    )


def run_min_synthetic_nodes_optimization(
    nodes_file: Path, cluster_file: Path, **kwargs
) -> bool:
    network_data = load_and_prepare_data(nodes_file, cluster_file, **kwargs)
    visualize_input_data(network_data)
    start_time = time.time()
    solver_result, status = solver_model_minimize_nodes(network_data)
    elapsed_time = time.time() - start_time
    visualize_model_output(network_data, solver_result, elapsed_time, cluster_file.stem)
    visualize_provider_utilization(cluster_file.stem, network_data, solver_result)

    # Print the output of the tool to preserve previous behavior
    print(solver_result["solver_output"])

    return status == "Optimal"


def run_swap_optimization(nodes_file: Path, cluster_file: Path, **kwargs) -> bool:
    network_data = load_and_prepare_data(nodes_file, cluster_file, **kwargs)

    scenario_name = Path(cluster_file).stem
    current_alloc_png = f"./output/current_node_allocation_{scenario_name}.png"
    visualize_current_node_allocation(
        network_data, scenario_name, output_path=current_alloc_png
    )

    start_time = time.time()
    swap_solver_result, status = solver_model_minimize_swaps(network_data)

    # Print the output of the tool to preserve previous behavior
    print(swap_solver_result["solver_output"])

    outcome = status == "Optimal"
    elapsed_time = time.time() - start_time

    json_file = f"./output/subnet_node_changes_{scenario_name}.json"
    png_file = f"./output/subnet_change_summary_{scenario_name}.png"

    export_subnet_node_changes_to_json(
        network_data, swap_solver_result, output_path=json_file
    )
    visualize_subnet_changes_from_json(
        scenario_name,
        elapsed_time=elapsed_time,
        json_path=json_file,
        output_path=png_file,
    )
    visualize_model_output(
        network_data, swap_solver_result, elapsed_time, cluster_file.stem
    )

    validate_subnet_assignment(
        node_file=nodes_file,
        topology_file=kwargs["topology_file"],
        json_file=json_file,
        pipeline_file=kwargs["node_pipeline_file"],
    )
    visualize_provider_utilization(cluster_file.stem, network_data, swap_solver_result)

    return outcome


@click.command()
@click.option(
    "--config-file",
    type=click.Path(exists=True, path_type=Path),
    default=Path("config.json"),
    show_default=True,
    help="Path to configuration JSON file",
)
def main(config_file: Path) -> None:
    """Run network topology optimizer using a configuration file."""
    with open(config_file) as f:
        config = json.load(f)

    # Validate config keys
    required_keys = [
        "nodes_file",
        "topology_file",
        "node_pipeline_file",
        "blacklist_file",
        "scenario",
        "mode",
        "no_synthetic_countries",
        "enforce_sev_constraint",
        "enforce_health_constraint",
    ]
    missing_keys = [k for k in required_keys if k not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")

    # Convert paths to Path objects
    nodes_file = Path(config["nodes_file"])
    topology_file = Path(config["topology_file"])
    node_pipeline_file = Path(config["node_pipeline_file"])
    blacklist_file = Path(config["blacklist_file"])
    scenario = Path(config["scenario"])

    if scenario.is_dir():
        scenario_files = sorted(scenario.glob("*.json"))
    else:
        scenario_files = [scenario]

    # Ensure output directory exists
    # Path("output").mkdir(exist_ok=True)

    # Optimization parameters
    mode = config["mode"]
    kwargs = dict(
        topology_file=topology_file,
        node_pipeline_file=node_pipeline_file,
        blacklist_file=blacklist_file,
        no_synthetic_countries=config["no_synthetic_countries"],
        enforce_sev_constraint=config["enforce_sev_constraint"],
        enforce_health_constraint=config["enforce_health_constraint"],
        enforce_blacklist_constraint=config["enforce_blacklist_constraint"],
        enforce_per_node_provider_assignation=config.get(
            "enforce_per_node_provider_assignation", False
        ),
        special_limits_file=config.get("special_limits_file", None),
    )

    outcomes = {}

    for scenario_path in scenario_files:
        click.echo(f"\n=== Running scenario: {scenario_path} ===")
        if mode != Mode.MIN_SYNTHETIC.value and mode != Mode.MIN_SWAPS.value:
            raise ValueError(f"Unknown mode: {mode}")

        try:
            if mode == Mode.MIN_SYNTHETIC.value:
                outcome = run_min_synthetic_nodes_optimization(
                    nodes_file, scenario_path, **kwargs
                )
            elif mode == Mode.MIN_SWAPS.value:
                outcome = run_swap_optimization(nodes_file, scenario_path, **kwargs)
        except Exception as e:
            print("".join(traceback.format_exception(type(e), e, e.__traceback__)))
            outcomes[scenario_path.name] = "❌ - Exception occured: " + str(e)
            continue

        outcomes[scenario_path.name] = (
            "✅ - Optimal solution found"
            if outcome
            else "❌ - Failed to find optimal solution"
        )

    print("Outcomes")
    print(json.dumps(outcomes, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
