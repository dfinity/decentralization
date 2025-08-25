"""
Small what-if tool that takes the output/*.json and applies it to the starting data/*.csv file

The goal is to use the next state as starting state to do additional checks and simulations of things going bad later.
"""

import pandas as pd
from topology_optimizer.utils import get_node_pipeline
import json

# This represents the file calculated by the tool that you want
# to apply to the network to do further testing from that new
# potential future starting point
INPUT_GENERATED_FILE = "./output/subnet_node_changes_21_all.json"

# This represents the file that you used to generate the above
# output. It was a starting point for the above result which
# should be adjusted.
# INPUT_NODES_CSV = "./data/current_nodes_20250623_140201.csv"
INPUT_NODES_CSV = "/home/nikola/Downloads/node_allocation_testing/base_case.csv"

# This represents the pipeline used to generate the output.
# They are needed to correctly finish the result. Keep in mind that
# for future runs after this script finishes you should remove
# all the nodes from the pipeline as they will be added to the
# new starting point.
INPUT_PIPELINE = "./data/node_pipeline.csv"

# This represents the destination where the changes will be written.
OUTPUT_NODES_CSV = "./output/resulting_nodes_distribution.csv"


def main():
    nodes_previous_df = pd.read_csv(INPUT_NODES_CSV)
    # Extract and remap pipeline nodes
    nodes_previous_pipeline = get_node_pipeline(INPUT_PIPELINE)
    nodes_previous_pipeline.rename(
        columns={
            "data_center": "dc_id",
            "node_provider": "node_provider_name",
            "data_center_provider": "owner",
        },
        inplace=True,
    )
    nodes_previous_pipeline["region"] = nodes_previous_pipeline["country"].map(
        lambda country: f",{country},"
    )

    nodes_previous_pipeline.drop(
        ["country", "is_sev", "node_operator", "is_synthetic", "is_available"],
        axis=1,
        inplace=True,
    )

    total_previous = pd.concat(
        [nodes_previous_df, nodes_previous_pipeline], ignore_index=True
    )

    with open(INPUT_GENERATED_FILE, "r") as f:
        changes = json.load(f)

    for subnet in changes:
        # First remove all of the nodes
        all_removed = [change["node_id"] for change in subnet["removed"]]
        total_previous.loc[total_previous["node_id"].isin(all_removed), "subnet_id"] = (
            ""
        )

    for subnet in changes:
        all_added = subnet["added"]
        added_node_ids = {change["node_id"] for change in all_added}

        # Identify existing and new nodes
        existing_mask = total_previous["node_id"].isin(added_node_ids)
        existing_node_ids = set(total_previous.loc[existing_mask, "node_id"])
        new_node_ids = added_node_ids - existing_node_ids

        # Update subnet_id for existing nodes
        total_previous.loc[existing_mask, "subnet_id"] = subnet["subnet_id"]

        # Prepare rows for new nodes
        new_rows = []
        for change in all_added:
            if change["node_id"] in new_node_ids:
                node_id = change["node_id"]
                region = (
                    "," + node_id.split("-")[1] + ","
                )  # Extract region from node_id
                new_rows.append(
                    {
                        "node_id": node_id,
                        "node_provider_name": change["node_provider"],
                        "status": "UP",
                        "node_type": "REPLICA",
                        "regions": region,
                        "subnet_id": subnet["subnet_id"],
                        "dc_id": region + "-dc",
                        "data_center_provider": region + "-dcp",
                        "node_operator": change["node_provider"],
                        "node_provider_id": change["node_provider"],
                    }
                )

        # Append new rows to total_previous
        if new_rows:
            total_previous = pd.concat(
                [total_previous, pd.DataFrame(new_rows)], ignore_index=True
            )

    total_previous.to_csv(OUTPUT_NODES_CSV, index=False)


if __name__ == "__main__":
    main()
