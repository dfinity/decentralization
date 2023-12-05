#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:17:00 2023

@author: bjoernassmann
"""

import pandas as pd
from helper_functions import create_node_dataframe, generate_synthetic_countries, generate_synthetic_nodes

def prepare_data(df_nodes, 
                 df_node_pipeline, 
                 df_candidate_nodes, 
                 network_topology, 
                 no_synthetic_countries, 
                 enforce_sev_constraint):
    # Preprocessing
    current_nodes = create_node_dataframe(df_nodes)
    
    # Adjust node_provider_name based on data_center values
    current_nodes.loc[current_nodes['data_center'].isin(['tp1', 'at1', 'fm1']), 'node_provider'] = "Aggregator: Palnu Logistics"
    current_nodes.loc[current_nodes['data_center'] == 'sj1', 'node_provider'] = "Aggregator: TH Glick"
    
    # Update node_provider_name for all nodes starting with "DFINITY" to just "DFINITY"
    current_nodes.loc[current_nodes['node_provider'].str.startswith('DFINITY'), 'node_provider'] = "DFINITY"

    # Combnine existing and candidate nodes
    current_nodes = pd.concat([current_nodes, df_node_pipeline, df_candidate_nodes], ignore_index=True)

    current_countries = list(current_nodes['country'].unique())
    synthetic_countries = generate_synthetic_countries(no_synthetic_countries)
    synthetic_nodes_df = generate_synthetic_nodes(synthetic_countries)
    node_all = pd.concat([current_nodes, synthetic_nodes_df], ignore_index=True)

    # Indices
    node_all_indices = node_all.index.tolist()
    synthetic_node_indicator = node_all['is_synthetic'].tolist()
    subnet_indices = network_topology.index.tolist()

    # node providers
    unique_node_providers = node_all['node_provider'].unique().tolist()
    node_providers_indices = list(range(len(unique_node_providers)))

    all_countries = current_countries + synthetic_countries
    all_countries_indices = list(range(len(all_countries)))
    
    # data center
    data_center_all = list(node_all['data_center'].unique())
    data_center_all_indices = list(range(len(data_center_all)))
    
    # data center provider
    data_center_provider_all = list(node_all['data_center_provider'].unique())
    data_center_provider_all_indices = list(range(len(data_center_provider_all)))
    
    network_data = {
        'network_topology': network_topology,
        'enforce_sev_constraint': enforce_sev_constraint,
        'subnet_indices': subnet_indices,
        'node_df': node_all,
        'node_indices': node_all_indices,
        'node_df_current': current_nodes,
        'node_df_synthetic': synthetic_nodes_df,
        'synthetic_node_indicator': synthetic_node_indicator,
        'node_provider_list': unique_node_providers,
        'node_provider_indices': node_providers_indices,
        'country_list': all_countries,
        'country_indices': all_countries_indices,
        'data_center_list': data_center_all,
        'data_center_indices': data_center_all_indices,
        'data_center_provider_list': data_center_provider_all,
        'data_center_provider_indices': data_center_provider_all_indices,
    }

    return network_data
    