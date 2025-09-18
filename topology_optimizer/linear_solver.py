#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MILP model for optimal node-to-subnet allocation in the ICP network.

"""

import logging
import tempfile

import pandas as pd
from pulp import (
    PULP_CBC_CMD,
    LpBinary,
    LpInteger,
    LpMinimize,
    LpProblem,
    LpStatus,
    LpVariable,
    lpSum,
    value,
)

from topology_optimizer.utils import get_special_limits, get_subnet_limit

# Standard attribute types to optimize over
ATTRIBUTE_NAMES = [
    "node_provider",
    "country",
    "data_center",
    "data_center_provider",
]


# Resembles the minimum number of nodes that the node provider
# has to have in order for the tool to ensure that at least one
# of his nodes is always assigned to a subnet.
MINIMUM_NODES_REQUIRED_FOR_ENSURING_PROVIDER_ASSIGNATION = 4

# ------------------------------------------------------------------------------
# INITIALIZATION
# ------------------------------------------------------------------------------


def init_lp_problem(network_data):
    """
    Initializes the LP problem and variables for nodes and attributes.
    Returns a dictionary of variables.
    """
    prob = LpProblem("Optimal_node_provider_distribution")

    node_indices = network_data["node_indices"]
    subnet_indices = network_data["subnet_indices"]

    node_allocations = LpVariable.dicts(
        "node_alloc", (node_indices, subnet_indices), cat=LpBinary
    )

    attribute_allocations = {}
    attribute_booleans = {}

    for attr in ATTRIBUTE_NAMES:
        indices = network_data[f"{attr}_indices"]
        attr_alloc = LpVariable.dicts(
            f"{attr}_alloc", (indices, subnet_indices), 0, None, LpInteger
        )
        attr_bool = LpVariable.dicts(
            f"{attr}_bool", (indices, subnet_indices), cat=LpBinary
        )
        attribute_allocations[attr] = attr_alloc
        attribute_booleans[attr] = attr_bool

    return {
        "prob": prob,
        "node_allocations": node_allocations,
        "attribute_allocations": attribute_allocations,
        "attribute_booleans": attribute_booleans,
    }


# ------------------------------------------------------------------------------
# CONSTRAINTS
# ------------------------------------------------------------------------------


def add_node_constraints(network_data, model):
    """
    Adds node constraints:
    - each node assigned at most once
    - subnet sizes matched
    - enforce SEV and availability if enabled
    """
    prob = model["prob"]
    node_alloc = model["node_allocations"]
    node_indices = network_data["node_indices"]
    subnet_indices = network_data["subnet_indices"]
    topology = network_data["network_topology"]
    node_df = network_data["node_df"]

    for node in node_indices:
        prob += (
            lpSum(node_alloc[node][subnet] for subnet in subnet_indices) <= 1,
            f"NodeOnce_{node}",
        )

    for subnet in subnet_indices:
        required = topology.loc[subnet, "subnet_size"]
        prob += (
            lpSum(node_alloc[node][subnet] for node in node_indices) == required,
            f"SubnetSize_{subnet}",
        )

    if network_data.get("enforce_sev_constraint", False):
        for subnet in subnet_indices:
            if topology.loc[subnet, "is_sev"]:
                for node in node_indices:
                    if not node_df.loc[node, "is_sev"]:
                        prob += (
                            node_alloc[node][subnet] == 0,
                            f"SEV_Block_{subnet}_{node}",
                        )

    if network_data.get("enforce_health_constraint", False):
        for subnet in subnet_indices:
            for node in node_indices:
                if not node_df.loc[node, "is_available"]:
                    prob += (
                        node_alloc[node][subnet] == 0,
                        f"Unavailable_Block_{subnet}_{node}",
                    )

    if network_data.get("enforce_blacklist_constraint", False):
        excluded_count = 0
        for subnet in subnet_indices:
            for node in node_indices:
                if node_df.loc[node, "is_blacklisted"]:
                    prob += (
                        node_alloc[node][subnet] == 0,
                        f"Blacklist_Block_{subnet}_{node}",
                    )
                    excluded_count += 1

    # Enforce per provider constraints
    if network_data.get("enforce_per_node_provider_assignation", False):
        add_per_node_provider_constraint(model, network_data)


def add_per_node_provider_constraint(model, network_data):
    """
    Adds the constraint for having at least one node per node provider assigned at all times.
    """
    prob = model["prob"]
    node_provider_list = network_data["node_provider_list_before_clustering"]
    node_df = network_data["node_df"]
    node_alloc = model["node_allocations"]
    subnet_indices = network_data["subnet_indices"]

    for provider in node_provider_list:
        provider_nodes = node_df[
            (node_df["original_node_provider"] == provider)
            & (~node_df["is_blacklisted"])
            & (node_df["is_available"])
        ].index

        # Skip the syntetic node provider.
        #
        # If they are used at all it means that they will already adhere to this rule
        # by being in some subnet.
        if provider.startswith("SYNTHETIC"):
            continue

        # As agreed, if a node provider is small, we cannot ensure that their nodes are
        # utilized at all times.
        if (
            len(provider_nodes)
            < MINIMUM_NODES_REQUIRED_FOR_ENSURING_PROVIDER_ASSIGNATION
        ):
            continue

        prob += (
            lpSum(
                node_alloc[node][subnet]
                for node in provider_nodes
                for subnet in subnet_indices
            )
            >= 1,
            f"MinOneNodeAssignedPerProvider_{provider}",
        )


def add_attribute_constraints(network_data, model, attr):
    """
    Links node_allocations to attribute_allocations for one attribute.
    """
    prob = model["prob"]
    node_alloc = model["node_allocations"]
    attr_alloc = model["attribute_allocations"][attr]
    attr_bool = model["attribute_booleans"][attr]

    node_df = network_data["node_df"]
    subnet_indices = network_data["subnet_indices"]
    attr_list = network_data[f"{attr}_list"]
    topology = network_data["network_topology"]

    for subnet in subnet_indices:
        subnet_size = topology.loc[subnet, "subnet_size"]
        for idx, val in enumerate(attr_list):
            nodes = node_df[node_df[attr] == val].index
            prob += (
                (
                    lpSum(node_alloc[n][subnet] for n in nodes)
                    == attr_alloc[idx][subnet]
                ),
                f"{attr}_Alloc_{val}_{subnet}",
            )
            prob += (
                (attr_alloc[idx][subnet] <= subnet_size * attr_bool[idx][subnet]),
                f"{attr}_BoolLink_{val}_{subnet}",
            )


def add_attribute_limits(network_data, model, attr):
    """
    Adds subnet-wise limits for attribute allocations.
    Applies special business rules for the NNS subnet.
    """
    prob = model["prob"]
    attr_alloc = model["attribute_allocations"][attr]
    subnet_indices = network_data["subnet_indices"]
    attr_indices = network_data[f"{attr}_indices"]
    attr_list = network_data[f"{attr}_list"]
    topology = network_data["network_topology"]

    network_special_limits = network_data["special_limits"]

    for subnet in subnet_indices:
        limit = get_subnet_limit(topology, subnet, attr)
        subnet_id = topology.loc[subnet, "subnet_id"]
        special_limits = get_special_limits(network_special_limits, subnet_id, attr)
        for idx in attr_indices:
            val = attr_list[idx]
            if val in special_limits:
                max_val, op = special_limits[val]
                if op == "eq":
                    prob += (
                        attr_alloc[idx][subnet] == max_val,
                        f"{attr}_{subnet}_Limit_eq_{val}",
                    )
                elif op == "lt":
                    prob += (
                        attr_alloc[idx][subnet] <= max_val,
                        f"{attr}_{subnet}_Limit_lt_{val}",
                    )
                elif op == "gt":
                    raise ValueError("`gt` doesn't make sense in our model.")


def add_attribute_subnet_allowed_values_constraints(
    network_data, model, attr, allowed_values, subnet_idx
):
    """
    Restricts a subnet to only accept nodes with specific attribute values.
    """
    prob = model["prob"]
    attr_alloc = model["attribute_allocations"][attr]
    attr_list = network_data[f"{attr}_list"]
    attr_indices = network_data[f"{attr}_indices"]

    allowed_idx = [attr_list.index(val) for val in allowed_values if val in attr_list]

    for idx in attr_indices:
        if idx not in allowed_idx:
            prob += (
                attr_alloc[idx][subnet_idx] == 0,
                f"{attr}_Disallow_{attr_list[idx]}_{subnet_idx}",
            )


def restrict_subnet_to_country_values(
    network_data, model, subnet_type, allowed_country_codes
):
    """
    Restricts a given subnet type to certain countries only.
    """
    topology = network_data["network_topology"]
    match = topology.index[topology["subnet_type"] == subnet_type]

    if match.empty:
        logging.warning(
            f"{subnet_type} not found in network_topology. Constraint not applied."
        )
        return  # Skip applying constraint

    subnet_idx = match[0]
    add_attribute_subnet_allowed_values_constraints(
        network_data, model, "country", allowed_country_codes, subnet_idx
    )


# ------------------------------------------------------------------------------
# PARSING RESULTS
# ------------------------------------------------------------------------------


def parse_results(network_data, model, verbose=False):
    """
    Extracts values from the solver and builds summary DataFrames.
    """
    prob = model["prob"]
    node_alloc = model["node_allocations"]
    node_df = network_data["node_df"]
    subnet_indices = network_data["subnet_indices"]
    node_indices = network_data["node_indices"]

    rows = []
    for node in node_indices:
        for subnet in subnet_indices:
            if value(node_alloc[node][subnet]) == 1:
                node_type = (
                    "synthetic" if node_df.loc[node, "is_synthetic"] else "current"
                )
                rows.append([node, subnet, node_type])

    df_node_alloc = pd.DataFrame(
        rows, columns=["node_index", "subnet_index", "node_type"]
    )
    model["df_node_allocations"] = df_node_alloc

    if verbose:
        print(df_node_alloc)

    for attr in ATTRIBUTE_NAMES:
        attr_list = network_data[f"{attr}_list"]
        attr_indices = network_data[f"{attr}_indices"]
        attr_alloc = model["attribute_allocations"][attr]

        rows = []
        for idx in attr_indices:
            for subnet in subnet_indices:
                val = value(attr_alloc[idx][subnet])
                if val > 0:
                    rows.append([attr_list[idx], subnet, val])

        df_attr = pd.DataFrame(rows, columns=[attr, "subnet_index", "allocation_value"])
        model[f"df_{attr}_allocations"] = df_attr

        if verbose:
            print(df_attr)

    if verbose:
        print("Status:", LpStatus[prob.status])

    return model


# ------------------------------------------------------------------------------
# SOLVER: MINIMIZE SYNTHETIC
# ------------------------------------------------------------------------------


def build_base_model(network_data):
    """
    Common logic to initialize and constrain the model.
    """
    model = init_lp_problem(network_data)

    add_node_constraints(network_data, model)

    for attr in ATTRIBUTE_NAMES:
        add_attribute_constraints(network_data, model, attr)
        add_attribute_limits(network_data, model, attr)

    # Country-specific subnet restrictions
    restrict_subnet_to_country_values(
        network_data,
        model,
        "European Subnet",
        [
            "AL",
            "AD",
            "AT",
            "BE",
            "BA",
            "BG",
            "HR",
            "CY",
            "CZ",
            "DK",
            "EE",
            "FI",
            "FR",
            "DE",
            "GR",
            "HU",
            "IS",
            "IE",
            "IT",
            "XK",
            "LV",
            "LI",
            "LT",
            "LU",
            "MT",
            "MD",
            "MC",
            "ME",
            "NL",
            "MK",
            "NO",
            "PL",
            "PT",
            "RO",
            "SM",
            "RS",
            "SK",
            "SI",
            "ES",
            "SE",
            "CH",
            "UA",
            "GB",
            "VA",
            "UK",
        ],
    )
    restrict_subnet_to_country_values(network_data, model, "Swiss Subnet", ["CH"])
    restrict_subnet_to_country_values(network_data, model, "US Subnet", ["US"])

    return model


def solver_model_minimize_nodes(network_data):
    """
    MILP solver to minimize synthetic nodes under all constraints.
    """
    model = build_base_model(network_data)
    prob = model["prob"]
    node_alloc = model["node_allocations"]

    is_synthetic = network_data["synthetic_node_indicator"]
    node_indices = network_data["node_indices"]
    subnet_indices = network_data["subnet_indices"]

    prob += (
        lpSum(
            node_alloc[i][j]
            for i in node_indices
            if is_synthetic[i]
            for j in subnet_indices
        ),
        "MinimizeSynthetic",
    )

    prob.sense = LpMinimize
    prob.objective_name = "minimize_node_swaps"

    with tempfile.NamedTemporaryFile() as t:
        prob.solve(PULP_CBC_CMD(msg=False, logPath=t.name))
        solver_output = t.read().decode()
    model["solver_output"] = solver_output

    return parse_results(network_data, model), LpStatus[prob.status]


def solver_model_minimize_swaps(network_data, min_synthetic_used=100):
    """
    MILP solver to minimize node reassignments (swaps) from an existing assignment,
    while satisfying all core constraints and limiting synthetic usage.
    """
    model = build_base_model(network_data)
    prob = model["prob"]
    node_alloc = model["node_allocations"]

    node_indices = network_data["node_indices"]
    subnet_indices = network_data["subnet_indices"]
    is_synthetic = network_data["synthetic_node_indicator"]

    # Synthetic usage constraint
    synthetic_used = lpSum(
        node_alloc[i][j]
        for i in node_indices
        if is_synthetic[i]
        for j in subnet_indices
    )
    prob += synthetic_used <= min_synthetic_used, "MaxSyntheticUsage"

    # Deviation setup
    current_assignment = network_data["current_assignment"]
    subnet_set = set(subnet_indices)

    # prev_assigned = [
    #     i
    #     for i in node_indices
    #     if not is_synthetic[i] and current_assignment[i] in subnet_set
    # ]
    # prev_unassigned = [i for i in node_indices if i not in prev_assigned]

    prev_assigned = []
    for i in range(len(current_assignment)):
        if not is_synthetic[i] and current_assignment[i] in subnet_set:
            prev_assigned.append(i)
    prev_unassigned = [i for i in node_indices if i not in prev_assigned]

    dropped = LpVariable.dicts("dropped", prev_assigned, 0, 1, cat=LpBinary)
    moved = LpVariable.dicts("moved", prev_assigned, 0, 1, cat=LpBinary)
    new_assign = LpVariable.dicts("new_assign", prev_unassigned, 0, 1, cat=LpBinary)

    for i in prev_assigned:
        old = current_assignment[i]
        prob += (
            dropped[i] >= 1 - lpSum(node_alloc[i][j] for j in subnet_indices),
            f"dropped_{i}",
        )
        prob += (
            moved[i] >= lpSum(node_alloc[i][j] for j in subnet_indices if j != old),
            f"moved_{i}",
        )

    for i in prev_unassigned:
        prob += (
            new_assign[i] >= lpSum(node_alloc[i][j] for j in subnet_indices),
            f"new_assign_{i}",
        )

    # Objective: weighted deviation
    # A dropped node and a newly assigned node (each counted separately in the objective function
    # with a cost of 1) can collectively be resolved with a single swap — replacing one node with another.
    # In contrast, a node move involves both removing the node from one subnet and placing it in another,
    # effectively requiring two swaps. To reflect this higher cost, we assign a weight of 2 to moves.
    # Note: The weighting formula could be fine tuned further.
    # For example the edge case of node A moving from subnet 1 to
    # subnet 2, and node B doing the inverse, could be done by 3 swaps.
    #
    # Moved nodes have additional deviation cost due to operational costs. Just dropping and adding nodes
    # can be done in one go, whereas moving the node from one subnet to another takes time for the proposals
    # to execute.
    deviation_cost = (
        lpSum(dropped[i] + (2 + 1) * moved[i] for i in prev_assigned)
        + lpSum(new_assign[i] for i in prev_unassigned)
        + 100 * synthetic_used
    )
    prob += deviation_cost
    prob.sense = LpMinimize
    prob.objective_name = "minimize_node_swaps"

    with tempfile.NamedTemporaryFile() as t:
        prob.solve(PULP_CBC_CMD(msg=False, logPath=t.name))
        solver_output = t.read().decode()
    model["solver_output"] = solver_output

    model = parse_results(network_data, model)

    # Track deviation counts
    def active(var_dict, indices, tol=1e-6):
        return [i for i in indices if value(var_dict[i]) > tol]

    model["deviation_nodes"] = {
        "dropped": active(dropped, prev_assigned),
        "moved": active(moved, prev_assigned),
        "new_assign": active(new_assign, prev_unassigned),
    }
    model["synthetic_used"] = value(synthetic_used)

    # Optional reporting
    print("Minimal node swaps to achieve target:")
    print(f"  → Dropped nodes       : {len(model['deviation_nodes']['dropped'])}")
    print(f"  → Moved nodes         : {len(model['deviation_nodes']['moved'])}")
    print(f"  → Newly assigned nodes: {len(model['deviation_nodes']['new_assign'])}")
    print(f"  → Synthetic nodes used: {model['synthetic_used']:.0f}")

    return model, LpStatus[prob.status]
