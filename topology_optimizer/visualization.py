#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pulp
import matplotlib.lines as mlines
from topology_optimizer.utils import calculate_nakamoto_coefficient
import pandas as pd
from itertools import cycle
import json
from pathlib import Path
from matplotlib.patches import Patch

def apply_chart_style(ax):
    """
    Apply clean chart formatting to a matplotlib axis.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="both", which="both", color="#999999", labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, color="#dddddd")
    ax.set_axisbelow(True)


def save_and_show_plot(output_path: str):
    """
    Save a matplotlib figure to a file, show it, then close it.
    """
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def generate_distinct_colors(n: int):
    """
    Generate up to 40 visually distinct colors using tab20 and tab20c.
    """
    cmap1 = plt.cm.tab20(np.arange(20))
    cmap2 = plt.cm.tab20c(np.arange(20))
    return [cmap1[i % 20] if i % 2 == 0 else cmap2[i % 20] for i in range(n)]


def prettify_dimension(name: str) -> str:
    """
    Convert snake_case dimension name to Title Case for display.
    """
    return name.replace("_", " ").title()


def visualize_current_node_allocation(
    network_data,
    scenario,
    output_path="./output/node_provider_CURRENT_node_allocation.png",
):
    """
    Plot CURRENT node-provider allocation per subnet.
    Highlights duplicate providers using distinct borders.
    Omits synthetic-only providers from the legend.
    """
    attribute = "node_provider"
    current_assign = network_data["current_assignment"]
    all_nodes_df = network_data["node_df"]
    node_df_syn = network_data.get("node_df_synthetic", pd.DataFrame())

    # This calculation works in the following steps:
    #  1. Create a temporary data frame containing all nodes (node_df_current)
    #     and only the synthetic nodes (node_df_syn).
    #  2. It adds node_df_syn twice to ensure that if they don't appear in already
    #     in all nodes (node_df_current) they will be mapped twice.
    #  3. It removes all of the duplicate rows. This will include any of the
    #     synthetic nodes that appear in all nodes (node_df_current) and
    #     synthetic nodes (node_df_syn).
    node_df_current = pd.concat(
        [all_nodes_df, node_df_syn, node_df_syn]
    ).drop_duplicates(keep=False)

    provider_list = network_data["node_provider_list"]

    # ------------------ Allocation table
    records = [
        (node_df_current.loc[i, attribute], subnet_idx)
        for i, subnet_idx in enumerate(current_assign)
        if subnet_idx is not None
    ]

    df_alloc = (
        pd.DataFrame(records, columns=[attribute, "subnet_index"])
        .value_counts()
        .reset_index(name="allocation_value")
    )

    pivot_df = (
        df_alloc.pivot(
            index="subnet_index", columns=attribute, values="allocation_value"
        )
        .fillna(0)
        .reindex(columns=provider_list, fill_value=0)
        .sort_index()
    )

    # ------------------ Color settings
    fill_colours = plt.cm.Blues(np.linspace(0.35, 0.85, len(provider_list)))
    fill_map = dict(zip(provider_list, fill_colours))

    synthetic_providers = (
        set(node_df_syn[attribute]) if not node_df_syn.empty else set()
    )
    
    duplicate_providers = {prov for prov in provider_list if (pivot_df[prov] > 1).any()}

    edge_palette = cycle(
        [
            "#e41a1c",
            "#984ea3",
            "#ff7f00",
            "#f781bf",
            "#ffff33",
            "#a65628",
            "#d95f02",
            "#1b9e77",
            "#4daf4a",
        ]
    )
    edge_map = {prov: next(edge_palette) for prov in duplicate_providers}

    # ------------------ Plot setup
    fig, ax = plt.subplots(figsize=(15, 10))
    bottoms = np.zeros(len(pivot_df))

    for prov in provider_list:
        values = pivot_df[prov]
        edge_cols = []

        for i, v in enumerate(values):
            should_mark_duplicate = False

            if prov in duplicate_providers and v > 1:
                # Special case: Don't mark DFINITY duplicates
                # This is allowed for the NNS subnet as per target topology.
                # We could fine tune this further by checking the subnet type.
                # Given that this is only a visualization, we'll keep it simple for now. 
                if prov == "DFINITY": 
                    should_mark_duplicate = False
                else:
                    should_mark_duplicate = True
            edge_cols.append(edge_map[prov] if should_mark_duplicate else "white")
        
        label = prov if prov not in synthetic_providers else "_nolegend_"
        
        ax.bar(
            pivot_df.index,
            values,
            bottom=bottoms,
            label=label,
            color=fill_map[prov],
            edgecolor=edge_cols,
            linewidth=1.2 if prov in duplicate_providers else 0.3,
        )
        bottoms += values

    # ------------------ Formatting and labels
    apply_chart_style(ax)

    ax.set_xlabel("Subnet Index", fontsize=11)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_title(
        "Current Node Allocations by Provider and Subnet",
        fontsize=14,
        fontweight="bold",
    )

    ax.text(
        0.65,
        0.95,
        f"Cluster scenario: {scenario}\n",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85
        ),
    )

    handles, labels = [], []
    for prov in provider_list:
        if prov in synthetic_providers:
            continue           # leave synthetic providers out of the legend

        face = fill_map[prov]
        edge = edge_map[prov] if prov in duplicate_providers and prov != "DFINITY" else "white"
        lw   = 1.2            if prov in duplicate_providers else 0.3
        handles.append(Patch(facecolor=face, edgecolor=edge, linewidth=lw))
        labels.append(prov)

    ax.legend(
        handles, labels,
        title="Node Providers",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=max(1, round(len(labels) / 35)),
        frameon=False,
        fontsize=9,
    )


    save_and_show_plot(output_path)


def visualize_derived_node_allocation(
    network_data,
    solver_result,
    nakamoto_result,
    elapsed_time,
    attribute,
    scenario_name,
):
    """
    Stacked bar chart for node allocations (optimized),
    by node_provider, data_center, etc. Includes Nakamoto coefficient overlays.
    """
    assert attribute in [
        "node_provider",
        "data_center",
        "data_center_provider",
        "country",
    ]
    pretty_attr = prettify_dimension(attribute)

    attribute_list = network_data[f"{attribute}_list"]
    node_df_current = network_data["node_df_current"]
    node_df_synthetic = network_data["node_df_synthetic"]
    df_allocations = solver_result[f"df_{attribute}_allocations"]
    prob = solver_result["prob"]
    nakamoto_coefficients = nakamoto_result[f"nakamoto_coefficients_{attribute}"]

    # Build color map (Blues for current, Reds for synthetic)
    current_attrs = list(node_df_current[attribute].unique())
    synthetic_attrs = (
        list(node_df_synthetic[attribute].unique())
        if not node_df_synthetic.empty
        else []
    )

    color_map = {
        **dict(
            zip(
                current_attrs, plt.cm.Blues(np.linspace(0.35, 0.85, len(current_attrs)))
            )
        ),
        **dict(
            zip(
                synthetic_attrs,
                plt.cm.Reds(np.linspace(0.35, 0.85, len(synthetic_attrs))),
            )
        ),
    }

    # Pivot to subnet-wise allocation matrix
    pivot_df = (
        df_allocations.pivot(
            index="subnet_index", columns=attribute, values="allocation_value"
        )
        .fillna(0)
        .reindex(columns=attribute_list, fill_value=0)
        .sort_index()
    )

    # Set up figure
    fig, ax = plt.subplots(figsize=(15, 10))
    bottoms = np.zeros(len(pivot_df))

    # Plot stacked bars
    for attr in attribute_list:
        values = pivot_df.get(attr, 0)
        ax.bar(
            pivot_df.index,
            values,
            bottom=bottoms,
            label=attr,
            color=color_map.get(attr, "#333333"),
            edgecolor="white",
            linewidth=0.5,
        )
        bottoms += values

    # Overlay Nakamoto coefficients
    for i, subnet_idx in enumerate(pivot_df.index):
        coeff = nakamoto_coefficients.get(subnet_idx, "")
        ax.text(i, bottoms[i] + 2, f"{coeff}", ha="center", fontsize=9)

    # Axis labels and title
    ax.set_xlabel("Subnet Index", fontsize=11)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_title(
        f"Optimized Node Allocation by {pretty_attr}", fontsize=14, weight="bold"
    )

    # Apply visual theme
    apply_chart_style(ax)

    # Legend
    ncol = max(1, round(len(attribute_list) / 35))
    ax.legend(
        title=f"{pretty_attr}s",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        ncol=ncol,
        frameon=False,
        fontsize=9,
    )

    # Annotate model meta info
    ax.text(
        0.70,
        0.95,
        f"Model: {prob.objective_name}\n"
        f"Wall Clock Time: {elapsed_time:.2f} sec\n"
        f"Objective Value: {pulp.value(prob.objective):.2f}\n"
        f"Total Nakamoto Coefficients: {sum(nakamoto_coefficients.values()):.2f}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85
        ),
    )

    # Save
    output_path = f"./output/{scenario_name}_{attribute}_node_allocation.png"
    save_and_show_plot(output_path)
    print(f"Saved plot to {output_path}")


def visualize_current_subnets(network_data):
    # Determine matrix width and height
    matrix_width = max(network_data["network_topology"]["subnet_size"])
    matrix_height = len(network_data["network_topology"]["subnet_size"])

    fig, ax = plt.subplots(figsize=(matrix_width / 4, matrix_height / 4))

    # Plot subnets
    for i, subnet_size in enumerate(network_data["network_topology"]["subnet_size"]):
        for j in range(subnet_size):
            ypos = j
            xpos = i
            cross_length = 0.2  # adjust to preference
            ax.plot(
                [xpos + 0.4 - cross_length, xpos + 0.4 + cross_length],
                [ypos + 0.4 - cross_length, ypos + 0.4 + cross_length],
                color="black",
            )
            ax.plot(
                [xpos + 0.4 - cross_length, xpos + 0.4 + cross_length],
                [ypos + 0.4 + cross_length, ypos + 0.4 - cross_length],
                color="black",
            )

    ax.set_xlim(
        0, matrix_height
    )  # matrix_height here as we are going by the number of subnets
    ax.set_ylim(
        0, matrix_width
    )  # matrix_width here as it represents the maximum number of nodes in a subnet
    ax.set_xlabel("Subnets")
    ax.set_ylabel("Number of nodes")
    ax.set_title("Slots of current subnets")

    # Legend
    cross_legend = mlines.Line2D(
        [0], [0], color="black", lw=0, marker="x", markersize=10, label="Subnet slot"
    )
    ax.legend(handles=[cross_legend], loc="upper right")

    plt.tight_layout()
    plt.show()


def visualize_node_topology_matrix(network_data, dimension, display_subnet_slots=True):
    """
    Visualize nodes by dimension (e.g., country, provider) in a matrix.
    Optionally overlays current subnet slots.
    """
    pretty_dim = prettify_dimension(dimension)
    df = network_data["node_df"][~network_data["node_df"]["is_synthetic"]]

    dim_counts = df[dimension].value_counts().sort_values(ascending=False)
    dim_values = dim_counts.index.tolist()
    colors = generate_distinct_colors(len(dim_values))
    color_map = dict(zip(dim_values, colors))

    max_width = 60
    matrix_width = min(dim_counts.iloc[0], max_width)
    max_subnet_size = max(network_data["network_topology"]["subnet_size"])
    matrix_height = max(max_subnet_size, len(dim_values))

    fig, ax = plt.subplots(figsize=(matrix_width / 4, matrix_height / 4))
    y_pos_counter = [0] * matrix_width

    for dim_val, count in dim_counts.items():
        shown_count = min(count, max_width)
        for j in range(shown_count):
            xpos = j
            ypos = y_pos_counter[j]
            ax.add_patch(
                plt.Rectangle(
                    (xpos, ypos), 0.7, 0.7, color=color_map[dim_val], linewidth=1
                )
            )
            y_pos_counter[j] += 1
        if count > max_width:
            ax.text(matrix_width + 1, y_pos_counter[shown_count - 1] - 1, str(count))

    if display_subnet_slots:
        for i, subnet_size in enumerate(
            network_data["network_topology"]["subnet_size"]
        ):
            for j in range(subnet_size):
                xpos = i
                ypos = j
                cross_len = 0.2
                ax.plot(
                    [xpos + 0.4 - cross_len, xpos + 0.4 + cross_len],
                    [ypos + 0.4 - cross_len, ypos + 0.4 + cross_len],
                    color="black",
                )
                ax.plot(
                    [xpos + 0.4 - cross_len, xpos + 0.4 + cross_len],
                    [ypos + 0.4 + cross_len, ypos + 0.4 - cross_len],
                    color="black",
                )

    ax.set_xlim(0, matrix_width)
    ax.set_ylim(0, matrix_height)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel(pretty_dim)
    ax.set_title(f"Node Topology Matrix by {pretty_dim}")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[val]) for val in dim_values]
    labels = dim_values.copy()

    if display_subnet_slots:
        cross_legend = mlines.Line2D(
            [0],
            [0],
            color="black",
            lw=0,
            marker="x",
            markersize=10,
            label="Subnet slot",
        )
        handles.append(cross_legend)
        labels.append("Subnet slot")

    ax.legend(
        handles, labels, title=pretty_dim, bbox_to_anchor=(1.05, 1), loc="upper left"
    )
    save_and_show_plot(f"./output/{dimension}_topology_matrix.png")


def visualize_node_topology_matrix_with_double_rows_per_country(network_data):
    """
    Visualize topology matrix with double-row spacing per country.
    """
    df = network_data["node_df"][~network_data["node_df"]["is_synthetic"]]
    country_counts = df["country"].value_counts().sort_values(ascending=False)
    countries = country_counts.index.tolist()
    colors = generate_distinct_colors(len(countries))
    color_map = dict(zip(countries, colors))

    matrix_width = min(country_counts.iloc[0] // 2, 60)
    max_subnet_size = max(network_data["network_topology"]["subnet_size"])
    matrix_height = max(max_subnet_size, len(countries) * 2)

    fig, ax = plt.subplots(figsize=(matrix_width / 4, matrix_height / 4))
    y_pos_counter = [0] * matrix_width

    for country, count in country_counts.items():
        capped_count = min(count, 100)
        for j in range(capped_count):
            xpos = j // 2
            ypos = y_pos_counter[j // 2] * 2 + j % 2
            ax.add_patch(
                plt.Rectangle(
                    (xpos, ypos), 0.8, 0.8, color=color_map[country], linewidth=1
                )
            )
            y_pos_counter[j // 2] += 1 if j % 2 == 1 else 0
        if count > 100:
            y_pos_label = y_pos_counter[(capped_count - 1) // 2] * 2 - 1
            ax.text(matrix_width + 1, y_pos_label, str(count), va="center")

    for i, subnet_size in enumerate(network_data["network_topology"]["subnet_size"]):
        for j in range(subnet_size):
            xpos = i
            ypos = j
            cross_len = 0.2
            ax.plot(
                [xpos + 0.4 - cross_len, xpos + 0.4 + cross_len],
                [ypos + 0.4 - cross_len, ypos + 0.4 + cross_len],
                color="black",
            )
            ax.plot(
                [xpos + 0.4 - cross_len, xpos + 0.4 + cross_len],
                [ypos + 0.4 + cross_len, ypos + 0.4 - cross_len],
                color="black",
            )

    ax.set_xlim(0, matrix_width)
    ax.set_ylim(0, matrix_height)
    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Country")
    ax.set_title("Node Topology Matrix by Country (Double Rows)")

    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[c]) for c in countries]
    cross_legend = mlines.Line2D(
        [0], [0], color="black", lw=0, marker="x", markersize=10, label="Subnet slot"
    )
    handles.append(cross_legend)
    countries.append("Subnet slot")

    ax.legend(
        handles,
        countries,
        title="Countries",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    save_and_show_plot("./output/country_double_row_topology_matrix.png")


def visualize_input_data(network_data):
    """
    Visualize initial network topology matrix data.

    """
    visualize_node_topology_matrix(network_data, "node_provider")
    visualize_node_topology_matrix(network_data, "data_center")
    visualize_node_topology_matrix(network_data, "data_center_provider")
    visualize_node_topology_matrix(network_data, "country")
    visualize_node_topology_matrix_with_double_rows_per_country(network_data)


def visualize_model_output(network_data, solver_result, elapsed_time, scenario):
    """
    Visualize the model's output - node allocation and Nakamoto coefficient.

    """
    nakamoto_result = calculate_nakamoto_coefficient(solver_result)
    for dimension in [
        "node_provider",
        "data_center",
        "data_center_provider",
        "country",
    ]:
        visualize_derived_node_allocation(
            network_data,
            solver_result,
            nakamoto_result,
            elapsed_time,
            dimension,
            scenario,
        )


def visualize_subnet_changes_from_json(
    scenario,
    json_path="./output/subnet_node_changes.json",
    output_path="./output/subnet_change_summary.png",
    elapsed_time=0.0,
):
    """
    Creates a vertical stacked bar chart showing
    node changes (unchanged, moved, added, dropped) across subnets.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    labels = []
    counts = {
        "unchanged": [],
        "moved_in": [],
        "newly_assigned": [],
        "newly_assigned_synthetic": [],
        "moved_away": [],
        "dropped": [],
    }

    totals = dict.fromkeys(counts, 0)

    for entry in data:
        labels.append(entry["subnet_type"] or entry["subnet_id"])

        def count_by_type(nodes, expected, *, synthetic=None):
            return sum(
                1
                for n in nodes
                if n.get("change_type") == expected
                and (
                    synthetic is None
                    or n.get("node_id", "").startswith("synthetic-") == synthetic
                )
            )

        unchanged = len(entry["unchanged"])
        moved_in = count_by_type(entry["added"], "moved_in")
        new_asg = count_by_type(entry["added"], "newly_assigned", synthetic=False)
        new_asg_syn = count_by_type(entry["added"], "newly_assigned", synthetic=True)
        moved_away = count_by_type(entry["removed"], "moved_away")
        dropped = count_by_type(entry["removed"], "dropped")

        counts["unchanged"].append(unchanged)
        counts["moved_in"].append(moved_in)
        counts["newly_assigned"].append(new_asg)
        counts["newly_assigned_synthetic"].append(new_asg_syn)
        counts["moved_away"].append(moved_away)
        counts["dropped"].append(dropped)

        totals["dropped"] += dropped
        totals["moved_away"] += moved_away
        totals["moved_in"] += moved_in
        totals["newly_assigned"] += new_asg + new_asg_syn

    total_removed = totals["dropped"] + totals["moved_away"]
    total_added = totals["moved_in"] + totals["newly_assigned"]
    total_swaps = max(total_removed, total_added)

    total_synthetic_new = sum(counts["newly_assigned_synthetic"])

    x_pos = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(max(14, len(labels) * 0.4), 8))

    palette = {
        "unchanged": "#d9d9d9",
        "moved_in": "#00a676",
        "newly_assigned": "#006ba6",
        "newly_assigned_synthetic": "#99ccff",
        "moved_away": "#f17c67",
        "dropped": "#8b0000",
    }

    pos_bottoms = [0] * len(labels)
    neg_bottoms = [0] * len(labels)

    def bar_segment(segment, label, positive=True):
        values = counts[segment] if positive else [-v for v in counts[segment]]
        bottoms = pos_bottoms if positive else neg_bottoms

        ax.bar(
            x_pos,
            values,
            bottom=bottoms,
            label=label,
            color=palette[segment],
            edgecolor="white",
            linewidth=0.3,
        )
        for i in range(len(bottoms)):
            bottoms[i] += values[i]

    # Stack segments
    bar_segment("dropped", "Dropped", positive=False)
    bar_segment("moved_away", "Moved away", positive=False)
    bar_segment("moved_in", "Moved in")
    bar_segment("newly_assigned_synthetic", "Newly assigned (synthetic)")
    bar_segment("newly_assigned", "Newly assigned (existing)")
    bar_segment("unchanged", "Unchanged")

    # Annotate positive additions
    for i in range(len(labels)):
        added = (
            counts["moved_in"][i]
            + counts["newly_assigned"][i]
            + counts["newly_assigned_synthetic"][i]
        )
        if added > 0:
            ax.text(
                x_pos[i],
                pos_bottoms[i] + 1,
                f"+{added}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#333333",
            )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Nodes", fontsize=11)
    ax.set_xlabel("Subnets", fontsize=11)
    ax.set_title("Node Reassignments by Subnet", fontsize=14, fontweight="bold")

    apply_chart_style(ax)

    leg = ax.legend(loc="upper right", frameon=False, title="Change Type", fontsize=9)
    leg.get_title().set_fontsize(14)

    ax.text(
        0.65,
        0.95,
        f"Model: Minimize node swaps\n"
        f"Elapsed time: {elapsed_time:.2f} sec\n"
        f"Cluster scenario: {scenario}\n"
        f"Total swaps: {total_swaps}\n"
        f"Total synthetic nodes: {total_synthetic_new}",
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="grey", alpha=0.85
        ),
    )

    save_and_show_plot(output_path)


def visualize_provider_utilization(scenario, network_data, solver_result):
    node_provider_list = network_data["node_provider_list"]
    node_df = network_data["node_df"]
    current_assignment = network_data["current_assignment"]
    final_df = solver_result["df_node_allocations"][["node_index", "subnet_index"]]
    final_assignment = dict(zip(final_df["node_index"], final_df["subnet_index"]))

    data = []

    for provider in node_provider_list:
        provider_mask = node_df["node_provider"] == provider
        provider_nodes = node_df[provider_mask].index

        before_util = 0
        after_util = 0

        for node in provider_nodes:
            # Synthetic providers don't have current utilization
            if not provider.startswith("SYNTHETIC_"):
                if (
                    len(current_assignment) > node
                    and current_assignment[node] is not None
                ):
                    before_util += 1

            if node in final_assignment and final_assignment[node] is not None:
                after_util += 1

        if before_util == 0 and after_util == 0 and provider.startswith("SYNTHETIC_"):
            continue

        api_bn_mask = node_df["node_type"] == "API_BOUNDARY"
        api_bn = len(node_df[provider_mask & api_bn_mask].index)

        data.append(
            {
                "provider": provider,
                "total_nodes": len(provider_nodes),
                "before_util": before_util,
                "after_util": after_util,
                "cordoned": max(
                    len(
                        node_df[
                            (node_df["node_provider"] == provider)
                            & (node_df["is_blacklisted"])
                        ].index
                    )
                    - api_bn,
                    0,
                ),
                "api_boundary_nodes": api_bn,
            }
        )

    data = sorted(data, key=lambda x: x["total_nodes"], reverse=True)

    def plot(ax, data):
        x = np.arange(len(data))
        width = 0.15

        offsets = [-2 * width, -width, 0, width, 2 * width]
        # Plot bars
        total_bar = ax.bar(
            x + offsets[0],
            [x["total_nodes"] for x in data],
            width,
            label="Total Nodes",
            color="white",
            edgecolor="black",
        )
        before_bar = ax.bar(
            x + offsets[1],
            [x["before_util"] for x in data],
            width,
            label="Current utilization",
            color="blue",
        )
        boundary_nodes = ax.bar(
            x + offsets[2],
            [x["api_boundary_nodes"] for x in data],
            width,
            label="API Boundary nodes",
            color="yellow",
        )
        blacklisted = ax.bar(
            x + offsets[3],
            [x["cordoned"] for x in data],
            width,
            label="Blacklisted",
            color="red",
        )
        after_bar = ax.bar(
            x + offsets[4],
            [x["after_util"] for x in data],
            width,
            label="After algorithm utilization",
            color="green",
        )

        ax.bar_label(total_bar, fontsize=20, padding=2)
        ax.bar_label(before_bar, fontsize=20, padding=2)
        ax.bar_label(boundary_nodes, fontsize=20, padding=2)
        ax.bar_label(after_bar, fontsize=20, padding=2)
        ax.bar_label(blacklisted, fontsize=20, padding=2)

        # Labels and titles
        ax.set_xlabel("Node Providers")
        ax.set_ylabel("Number of Nodes")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [x["provider"] for x in data], rotation=45, ha="right", fontsize=22
        )

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(max(12, len(node_provider_list) // 2), 40)
    )

    amount = len(data) // 4
    plot(ax1, data[0:amount])
    plot(ax2, data[amount : 2 * amount])
    plot(ax3, data[2 * amount : 3 * amount])
    plot(ax4, data[3 * amount :])

    ax1.set_title("Node Utilization by Provider")
    ax1.legend(
        fontsize=20,
        ncol=1,
    )

    plt.tight_layout()
    save_and_show_plot(f"./output/{scenario}_per_provider_utilization.png")
