#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:15:51 2023

@author: bjoernassmann
"""

import pandas as pd


###############################################################################
# Helper functions 
def create_dataframe(node_list):
    df = pd.DataFrame(node_list)
    df.columns = ['node_provider_name', 'no_nodes', 'data_center', 'data_center_provider', 'country']
    df['is_synthetic'] = False
    return df

def generate_synthetic_countries(no_synthetic_countries):
    return ['C' + str(i+1) for i in range(no_synthetic_countries)]

def is_sev_node_provider(node_provider):
    # List of specific node providers
    sev_providers = [
        "Accuset Solutions",
        "Anonstake",
        "Anypoint Pty Ltd",
        "Aspire Properties",
        "Coplus Limited",
        "Exaion",
        "Geodd Pvt Ltd",
        "George Bassadone",
        "Honeycomb Capital (Pty) Ltd",
        "Icaria Systems Pty Ltd",
        "Illusions In Art (Pty) Ltd",
        "InfoObjects",
        "Karel Frank",
        "Lukas Helebrandt",
        "Marc Johnson",
        "Marvelous Web3",
        "MB Patrankos šūvis",
        "NoviSystems, LLC",
        "Pindar Technology Limited",
        "Rivram Inc",
        "Wancloud limited",
        "Wolkboer (Pty) Ltd",
        "Zondax AG"
    ]
    
    # Check if node_provider is in sev_providers
    return node_provider in sev_providers


def create_node_dataframe(df_nodes):
    # Initialize an empty list to hold the new records
    new_records = []
    
    # Iterate through each row in the existing DataFrame
    for idx, row in df_nodes.iterrows():
        # Extract existing information
        node_provider = row['node_provider_name']
        dc_id = row['dc_id']  
        owner = row['owner'] 
        region = row['region'].split(",")[1]  
        is_sev = is_sev_node_provider(node_provider)
        
        # Create a new record with the desired fields
        new_record = {
            "node_provider": node_provider,
            "data_center": dc_id,  
            "data_center_provider": owner,  
            "country": region,
            "is_synthetic": False,  
            "is_sev":  is_sev
        }
        
        # Append the new record to the list
        new_records.append(new_record)
    
    # Create a new DataFrame using the list of new records
    current_nodes_new = pd.DataFrame(new_records)
    
    return current_nodes_new

def generate_synthetic_nodes( synthetic_countries, no_node_provider_per_country=5, no_nodes_per_provider=4):
    synthetic_nodes = ['NP' + str(i+1) for i in range(no_node_provider_per_country * len(synthetic_countries))]
    synthetic_dc = ['DC' + str(i+1) for i in range(no_node_provider_per_country * len(synthetic_countries))]
    synthetic_dc_provider = ['DCP' + str(i+1) for i in range(no_node_provider_per_country * len(synthetic_countries))]
    data = []
    provider_index = 0
    for country in synthetic_countries:
        for _ in range(no_node_provider_per_country):
            # simplyfing assumption: each node provider has own dc and dc provider
            provider_name = synthetic_nodes[provider_index]
            dc_name = synthetic_dc[provider_index]
            dcp_name = synthetic_dc_provider[provider_index]
            for _ in range(no_nodes_per_provider):
                new_record = {
                    'node_provider': provider_name,
                    'data_center': dc_name,
                    'data_center_provider': dcp_name,
                    'country': country,
                    'is_synthetic': True,
                    "is_sev": True
                    }
                data.append(new_record)
            provider_index += 1
    return pd.DataFrame(data)


def get_network_config(df_subnet):
    network_config_list = []
    
    # Iterate through each row in the DataFrame
    for _, s in df_subnet.iterrows():
        size = s['total_nodes']
        nakamoto_node_provider = round(size / 3 + 1)
        
        if nakamoto_node_provider == 5:
            nakamoto_country = 3
        else:
            nakamoto_country = 4
            
        network_config_list.append((size, nakamoto_node_provider, nakamoto_country))

    return pd.DataFrame(network_config_list, columns=['subnet_size', 'nakamoto_target_node_provider', 'nakamoto_target_country'])


def calculate_nakamoto_for_attribute(df_attribute_allocations):
    nakamoto_coefficients = {}
    subnets_indices = df_attribute_allocations['subnet_index'].unique()
    
    for subnet_idx in subnets_indices:
        subnet_entries = df_attribute_allocations[df_attribute_allocations['subnet_index'] == subnet_idx]
        if not subnet_entries.empty:
            total_nodes_in_subnet = subnet_entries['allocation_value'].sum()
            sorted_entries = subnet_entries.sort_values(by='allocation_value', ascending=False)

            sum_nodes = 0
            nakamoto_coefficient = 0
            for _, row in sorted_entries.iterrows():
                sum_nodes += row['allocation_value']
                nakamoto_coefficient += 1
                if sum_nodes > total_nodes_in_subnet / 3:
                    break
            nakamoto_coefficients[subnet_idx] = nakamoto_coefficient
        else:
            nakamoto_coefficients[subnet_idx] = 0  # No attributes allocated in this subnet

    return nakamoto_coefficients

def calculate_nakamoto_coefficient(result):
    attributes = ["node_provider", "data_center", "data_center_provider", "country"]
    nakamoto_result = {}
    
    for attribute in attributes:
        df_attribute_allocations = result[f'df_{attribute}_allocations']
        nakamoto_coefficients = calculate_nakamoto_for_attribute(df_attribute_allocations)
        nakamoto_result[f'nakamoto_coefficients_{attribute}'] = nakamoto_coefficients

    return nakamoto_result


def get_target_topology(subnet_limits):
    # Step 1: Create a DataFrame with the static data
    data = {
        "subnet_type": ["NNS", "SNS", "Fiduciary", "II", "ECDSA signing", "ECDSA backup",
                 "Bitcoin canister", "European Subnet", "Swiss Subnet"] + ["Application"] * 31,
        "subnet_size": [43, 34, 28, 28, 28, 28, 13, 13, 13] + [13] * 31,
        "is_sev": [False, False, False, True, True, True, False, True, True] + [False] * 31
    }
    df = pd.DataFrame(data)
    
    # Step 2: Populate DataFrame with subnet_limit_xxx and nakamoto_xxx columns
    attribute_list = ["node_provider", "data_center", "data_center_provider", "country"]
    
    for attr in attribute_list:
        subnet_limit_col_name = f"subnet_limit_{attr}"
        nakamoto_col_name = f"nakamoto_target_{attr}"
        
        # Add subnet_limit column based on provided subnet_limits record
        df[subnet_limit_col_name] = subnet_limits[attr]
        
        # Add nakamoto column based on the provided formula
        df[nakamoto_col_name] = (df["subnet_size"] // (3 * df[subnet_limit_col_name])) + 1
        
    return df

def get_Sep23_topology(subnet_limits):
    # Step 1: Create a DataFrame with the static data
    data = {
        "subnet_type": ["NNS", "SNS", "Fiduciary (also ECDSA signing)", "II",
                 "Bitcoin canister"] + ["Application"] * 31,
        "subnet_size": [40, 34, 28, 28, 13] + [13] * 31,
        "is_sev": [False, False, True, True, False] + [False] * 31
    }
    df = pd.DataFrame(data)
    
    # Step 2: Populate DataFrame with subnet_limit_xxx and nakamoto_xxx columns
    attribute_list = ["node_provider", "data_center", "data_center_provider", "country"]
    
    for attr in attribute_list:
        subnet_limit_col_name = f"subnet_limit_{attr}"
        nakamoto_col_name = f"nakamoto_target_{attr}"
        
        # Add subnet_limit column based on provided subnet_limits record
        df[subnet_limit_col_name] = subnet_limits[attr]
        
        # Add nakamoto column based on the provided formula
        df[nakamoto_col_name] = (df["subnet_size"] // (3 * df[subnet_limit_col_name])) + 1
        
    return df

def get_subnet_limit(network_topology, subnet_index, attribute):
    column_name = f"subnet_limit_{attribute}"
    return network_topology.at[subnet_index, column_name]

def get_nakamoto_target(network_topology, subnet_index, attribute):
    column_name = f"nakamoto_target_{attribute}"
    return network_topology.at[subnet_index, column_name]

def create_candidate_node_dataframe(node_provider,
                                    data_center,
                                    data_center_provider,
                                    country,
                                    is_sev,
                                    no_nodes):
    new_records = []
    for _ in range(no_nodes):
        # Create a new record with the desired fields
        new_record = {
            'node_provider': node_provider,
            'data_center': data_center,
            'data_center_provider': data_center_provider,
            'country': country,
            'is_synthetic': False,
            "is_sev": is_sev
            }
        
        # Append the new record to the list
        new_records.append(new_record)
    
    # Create a new DataFrame using the list of new records
    candidate_nodes = pd.DataFrame(new_records)
    
    return candidate_nodes


