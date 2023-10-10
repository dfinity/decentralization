#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:18:26 2023

@author: bjoernassmann
"""

from helper_functions import get_subnet_limit, get_nakamoto_target
from pulp import LpProblem, LpVariable, LpInteger, LpMaximize, LpStatus, lpSum, LpMinimize
import pandas as pd
from itertools import combinations


def init_lp_problem(network_data):
    result = {}
    # Extract needed variables from network_data
    node_indices = network_data['node_indices']
    subnet_indices = network_data['subnet_indices']
    node_provider_indices = network_data['node_provider_indices']
    country_indices = network_data['country_indices']
    data_center_indices = network_data['data_center_indices']
    data_center_provider_indices = network_data['data_center_provider_indices']
    
    
    # Define the problem
    prob = LpProblem("Optimal_node_provider_distribution", LpMaximize)

    # Variable for node allocation
    node_allocations = LpVariable.dicts("node_allocations", (node_indices, subnet_indices), cat='Binary')

    # Variable for node provider allocation
    node_provider_allocations = LpVariable.dicts("node_provider_allocations", (node_provider_indices, subnet_indices), 0, None, LpInteger)
    node_provider_allocations_boolean = LpVariable.dicts("node_provider_allocations_boolean", (node_provider_indices, subnet_indices), cat='Binary')

    # Variable for country allocation
    country_allocations = LpVariable.dicts("country_allocations", (country_indices, subnet_indices), 0, None, cat='Integer')
    country_allocations_boolean = LpVariable.dicts("country_allocations_boolean", (country_indices, subnet_indices), cat='Binary')
    
    # Variable for data center allocation
    data_center_allocations = LpVariable.dicts("data_center_allocations", (data_center_indices, subnet_indices), 0, None, cat='Integer')
    data_center_allocations_boolean = LpVariable.dicts("data_center_allocations_boolean", (data_center_indices, subnet_indices), cat='Binary')
    
    # Variable for data center provider allocation
    data_center_provider_allocations = LpVariable.dicts("data_center_provider_allocations", (data_center_provider_indices, subnet_indices), 0, None, cat='Integer')
    data_center_provider_allocations_boolean = LpVariable.dicts("data_center_provider_allocations_boolean", (data_center_provider_indices, subnet_indices), cat='Binary')

    result.update({
        'prob': prob,
        'node_allocations': node_allocations,
        'node_provider_allocations': node_provider_allocations,
        'node_provider_allocations_boolean': node_provider_allocations_boolean,
        'country_allocations': country_allocations,
        'country_allocations_boolean': country_allocations_boolean,
        'data_center_allocations': data_center_allocations,
        'data_center_allocations_boolean': data_center_allocations_boolean,
        'data_center_provider_allocations': data_center_provider_allocations,
        'data_center_provider_allocations_boolean': data_center_provider_allocations_boolean,
    })

    return result

def add_node_constraints(network_data, result):
    # Extract needed variables
    prob = result['prob']
    node_allocations = result['node_allocations']
    node_indices = network_data['node_indices']
    subnet_indices = network_data['subnet_indices']
    network_topology = network_data['network_topology']
    node_df = network_data['node_df'] 
    enforce_sev_constraint = network_data['enforce_sev_constraint']
    
    # Constraint: Each node is allocated only once
    for node_idx in node_indices:
        prob += lpSum([node_allocations[node_idx][subnet_idx] for subnet_idx in subnet_indices]) <= 1, f"NodeOnce_{node_idx}"

    # Constraint: Each subnet gets nodes according to its size
    for subnet_idx in subnet_indices:
        subnet_size = network_topology.loc[subnet_idx, 'subnet_size']
        prob += lpSum([node_allocations[node_idx][subnet_idx] for node_idx in node_indices]) == subnet_size, f"SubnetSize_{subnet_idx}"
    
    # Constraint: If a subnet is SEV, nodes should be SEV
    if enforce_sev_constraint:
        for subnet_idx in subnet_indices:
            if network_topology.loc[subnet_idx, 'is_sev']:
                for node_idx in node_indices:
                    # Ensure that non-SEV nodes are not assigned to an SEV subnet
                    if not node_df.loc[node_idx, 'is_sev']:
                        prob += node_allocations[node_idx][subnet_idx] == 0, f"SEV_Subnet_{subnet_idx}_Node_{node_idx}"

def add_attribute_allocations_constraints(network_data, result, attribute):
    node_df = network_data['node_df']
    subnet_indices = network_data['subnet_indices']
    attribute_list = network_data[f"{attribute}_list"]
    network_topology = network_data['network_topology']

    prob = result['prob']
    node_allocations = result['node_allocations']
    attribute_allocations = result[f"{attribute}_allocations"]
    attribute_allocations_boolean = result[f"{attribute}_allocations_boolean"]

    # Link node_allocations and attribute_allocations
    for subnet_idx in subnet_indices:
        subnet_size = network_topology.loc[subnet_idx, 'subnet_size']  
        for attribute_idx, attribute_instance in enumerate(attribute_list):
            relevant_nodes = node_df[node_df[attribute] == attribute_instance].index.tolist()
            prob += lpSum([node_allocations[node_idx][subnet_idx] for node_idx in relevant_nodes]) == attribute_allocations[attribute_idx][subnet_idx], f"{attribute}_Allocation_{attribute_instance}_Subnet_{subnet_idx}"
            prob += attribute_allocations[attribute_idx][subnet_idx] >= attribute_allocations_boolean[attribute_idx][subnet_idx], f"BooleanLink1_{attribute}_{attribute_idx}_{subnet_idx}"
            prob += attribute_allocations[attribute_idx][subnet_idx] <= subnet_size * attribute_allocations_boolean[attribute_idx][subnet_idx], f"BooleanLink2_{attribute}_{attribute_idx}_{subnet_idx}"


def add_attribute_subnet_limit_constraints(network_data, result, attribute):
    subnet_indices = network_data['subnet_indices']
    attribute_indices = network_data[f"{attribute}_indices"]
    network_topology = network_data['network_topology']
    attribute_allocations = result[f"{attribute}_allocations"]
    prob = result['prob']

    for subnet_idx in subnet_indices:
        attribute_subnet_limit = get_subnet_limit(network_topology, subnet_idx, attribute)
        for attribute_idx in attribute_indices:
            prob += attribute_allocations[attribute_idx][subnet_idx] <= attribute_subnet_limit, f"MaxNodesPerAttribute_{attribute}_{attribute_idx}_Subnet_{subnet_idx}"
   

def add_attribute_nakamoto_constraints(network_data, result, attribute):
    subnet_indices = network_data['subnet_indices']
    attribute_indices = network_data[f"{attribute}_indices"]
    network_topology = network_data['network_topology']
    attribute_allocations = result[f"{attribute}_allocations"]
    prob = result['prob']    
    
    for subnet_idx in subnet_indices:
        subnet_size = network_topology.loc[subnet_idx, 'subnet_size']
        nakamoto_target = get_nakamoto_target(network_topology, subnet_idx, attribute)

        # Go through all combinations of k-1 attribute instances
        for attribute_subsets in combinations(attribute_indices, nakamoto_target - 1):
            epsilon = 1e-8
            prob += lpSum([attribute_allocations[attribute_idx][subnet_idx] for attribute_idx in attribute_subsets]) + epsilon <= (1 / 3) * subnet_size, f"Nakamoto_{subnet_idx}_{attribute_subsets}"
 


def parse_results(network_data, result):
    attributes = ["node_provider", "data_center", "data_center_provider", "country"]
    prob = result['prob']

    # Extracting node_allocations values into a dictionary
    node_indices = network_data['node_indices']
    subnet_indices = network_data['subnet_indices']
    node_allocations = result['node_allocations']
    node_df = network_data['node_df']

    node_allocations_values = {(node_idx, subnet_idx): node_allocations[node_idx][subnet_idx].varValue
                               for node_idx in node_indices for subnet_idx in subnet_indices}

    # Convert the dictionary to a Pandas DataFrame for easier inspection
    rows = []
    for (node_idx, subnet_idx), value in node_allocations_values.items():
        if value == 1:  # Only consider non-empty allocations
            node_type = node_df['is_synthetic'].iloc[node_idx]
            node_type_str = 'synthetic' if node_type else 'current'
            rows.append([node_idx, subnet_idx, node_type_str, value])

    df_node_allocations = pd.DataFrame(rows, columns=['Node_Index', 'Subnet_Index', 'Node_Type', 'Allocation_Value'])
    print(df_node_allocations)

    # Looping over attributes to extract, convert and print their allocations
    for attribute in attributes:
        attribute_list = network_data[f"{attribute}_list"]
        attribute_indices = network_data[f"{attribute}_indices"]
        attribute_allocations = result[f"{attribute}_allocations"]

        attribute_allocations_values = {(attribute_idx, subnet_idx): attribute_allocations[attribute_idx][subnet_idx].varValue
                                        for attribute_idx in attribute_indices for subnet_idx in subnet_indices}

        attr_rows = []
        for (attribute_idx, subnet_idx), value in attribute_allocations_values.items():
            if value > 0:  # Only consider non-empty allocations
                attribute_name = attribute_list[attribute_idx]
                attr_rows.append([attribute_name, subnet_idx, value])

        df_attribute_allocations = pd.DataFrame(attr_rows, columns=[f'{attribute}', 'subnet_index', 'allocation_value'])
        print(df_attribute_allocations)

        # Store these dataframes in the result dictionary
        result[f'df_{attribute}_allocations'] = df_attribute_allocations

    print("Status:", LpStatus[prob.status])

    # Storing the node allocation DataFrame in the result dictionary
    result['df_node_allocations'] = df_node_allocations

    return result  


def define_objective_function_maximize_country_decentralization_by_node_set(network_data, result):
    subnet_indices = network_data['subnet_indices']
    country_indices = network_data['country_indices']
    
    prob = result['prob']

    country_allocations_boolean = result['country_allocations_boolean']
    
    prob += lpSum([country_allocations_boolean[country_idx][subnet_idx] 
                  for country_idx in country_indices 
                  for subnet_idx in subnet_indices]), "Maximize_Country_Allocation"
    prob.sense = LpMaximize 
    prob.objective_name = "Maximize_Country_Allocations"


def solver_model_maximize_country_decentralization_by_node_set(network_data):
    result = {}
    
    # Initialize the LP problem
    result.update(init_lp_problem(network_data))
    
    # Add constraints
    add_node_constraints(network_data, result)
    
    add_attribute_allocations_constraints(network_data, result, 'node_provider')
    add_attribute_allocations_constraints(network_data, result, 'data_center')
    add_attribute_allocations_constraints(network_data, result, 'data_center_provider')
    add_attribute_allocations_constraints(network_data, result, 'country')
    
    add_attribute_subnet_limit_constraints(network_data, result, 'node_provider')
    add_attribute_subnet_limit_constraints(network_data, result, 'data_center')
    add_attribute_subnet_limit_constraints(network_data, result, 'data_center_provider')
    add_attribute_subnet_limit_constraints(network_data, result, 'country')
    
    define_objective_function_maximize_country_decentralization_by_node_set(network_data, result)
    
    # Solve the LP problem
    result['prob'].solve()
    
    # Parse results
    result = parse_results(network_data, result)
    
    return result


def define_objective_function_model_minimize_nodes_by_diversification_target(network_data, result):
    node_indices = network_data['node_indices']
    subnet_indices = network_data['subnet_indices']
    synthetic_node_indicator = network_data['synthetic_node_indicator']
    prob = result['prob']
    node_allocations = result['node_allocations']
    
    prob += lpSum([node_allocations[node_idx][subnet_idx] * (1 if synthetic_node_indicator[node_idx] else 0) 
                   for node_idx in node_indices for subnet_idx in subnet_indices]), "Minimize_Synthetic"
    prob.sense = LpMinimize
    prob.objective_name = "Minimize_Synthetic_Nodes_By_Diversification_Target"



def solver_model_minimize_nodes_by_subnet_limit(network_data):
    result = {}
    
    # Initialize the LP problem
    result.update(init_lp_problem(network_data))
    
    # Add constraints
    add_node_constraints(network_data, result)
    
    add_attribute_allocations_constraints(network_data, result, 'node_provider')
    add_attribute_allocations_constraints(network_data, result, 'data_center')
    add_attribute_allocations_constraints(network_data, result, 'data_center_provider')
    add_attribute_allocations_constraints(network_data, result, 'country')
    
    add_attribute_subnet_limit_constraints(network_data, result, 'node_provider')
    add_attribute_subnet_limit_constraints(network_data, result, 'data_center')
    add_attribute_subnet_limit_constraints(network_data, result, 'data_center_provider')
    add_attribute_subnet_limit_constraints(network_data, result, 'country')
        
    # define objective
    define_objective_function_model_minimize_nodes_by_diversification_target(network_data, result)
    
    # Solve the LP problem
    result['prob'].solve()
    
    # Parse results
    result = parse_results(network_data, result)  
    
    return result

def solver_model_minimize_nodes_by_country_nakamoto_target(network_data):
    result = {}
    
    # Initialize the LP problem
    result.update(init_lp_problem(network_data))
    
    # Add constraints
    add_node_constraints(network_data, result)
    
    add_attribute_allocations_constraints(network_data, result, 'node_provider')
    add_attribute_allocations_constraints(network_data, result, 'data_center')
    add_attribute_allocations_constraints(network_data, result, 'data_center_provider')
    add_attribute_allocations_constraints(network_data, result, 'country')
    
    add_attribute_subnet_limit_constraints(network_data, result, 'node_provider')
    add_attribute_subnet_limit_constraints(network_data, result, 'data_center')
    add_attribute_subnet_limit_constraints(network_data, result, 'data_center_provider')
    add_attribute_nakamoto_constraints(network_data, result, 'country')
        
    # define objective
    define_objective_function_model_minimize_nodes_by_diversification_target(network_data, result)
    
    # Solve the LP problem
    result['prob'].solve()
    
    # Parse results
    result = parse_results(network_data, result)  
    
    return result