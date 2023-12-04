#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Network Topology Analysis and Visualization Script.

This script processes and visualizes the network topology based on node data
and predefined subnet limits, applies a solver model for optimal node allocation, and visualizes
the resulting node allocation and Nakamoto coefficients on a subnet level.

@author: bjoernassmann
"""

import time
from data_preparation import prepare_data
from linear_solver import solver_model_minimize_nodes_by_subnet_limit
from visualization import (
    visualize_node_topology_matrix,
    visualize_node_allocation,
    visualize_node_topology_matrix_with_double_rows_per_country
)
from helper_functions import get_target_topology, calculate_nakamoto_coefficient, create_candidate_node_dataframe


def visualize_input_data(network_data):
    """
    Visualize initial network topology matrix data.

    Parameters:
        network_data (dict): Preprocessed network data.
    """
    visualize_node_topology_matrix(network_data, 'node_provider')
    visualize_node_topology_matrix(network_data, 'data_center') 
    visualize_node_topology_matrix(network_data, 'data_center_provider') 
    visualize_node_topology_matrix(network_data, 'country') 
    visualize_node_topology_matrix_with_double_rows_per_country(network_data)


def visualize_model_output(network_data, solver_result, elapsed_time):
    """
    Visualize the model's output - node allocation and Nakamoto coefficient.

    Parameters:
        network_data (dict): Preprocessed network data.
        solver_result (dict): Output from the solver model.
        elapsed_time (float): Time taken by the solver model.
    """
    nakamoto_result = calculate_nakamoto_coefficient(solver_result)
    for dimension in ['node_provider', 'data_center', 'data_center_provider', 'country']:
        visualize_node_allocation(
            network_data, solver_result, nakamoto_result, elapsed_time, dimension
        )


def main():
    """Main function to execute the analysis and visualization pipeline."""
    # Step 1: Input data
    from ic_topology_20230907 import df_nodes  # Loading node population data
    df_candidate_nodes = create_candidate_node_dataframe(node_provider ='Lionel Messi',
                                                         data_center ='Buenos Aires 1',
                                                         data_center_provider ='Peron Corporation',
                                                         country = 'AR',
                                                         is_sev = True,
                                                         no_nodes = 0)

    subnet_limits = {
        "node_provider": 1,
        "data_center": 1,
        "data_center_provider": 1,
        "country": 2
    }
    network_topology = get_target_topology(subnet_limits)
    network_topology.loc[network_topology['subnet_size'] >= 28, 'subnet_limit_country'] = 3

    network_data = prepare_data( df_nodes, 
                                df_candidate_nodes,
                                network_topology,
                                no_synthetic_countries=12, 
                                enforce_sev_constraint=True
    )
    
    # Step 2: Display input data
    visualize_input_data(network_data)
    
    # Step 3: Apply solver
    start_time = time.time()
    solver_result = solver_model_minimize_nodes_by_subnet_limit(network_data)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Step 4: Display model output
    visualize_model_output(network_data, solver_result, elapsed_time)


if __name__ == "__main__":
    main()
