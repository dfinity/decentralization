#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:26:08 2023

@author: bjoernassmann
"""

import matplotlib.pyplot as plt
import numpy as np
import pulp
import matplotlib.lines as mlines
import os



def visualize_node_allocation(network_data, solver_result, nakamoto_result, elapsed_time, attribute):
    assert attribute in ["node_provider", "data_center", "data_center_provider", "country"], "Invalid attribute provided."

    # Extract relevant information from the input records
    attribute_list = network_data[f'{attribute}_list']
    node_df_current = network_data['node_df_current']
    node_df_synthetic = network_data['node_df_synthetic']
    df_allocations = solver_result[f'df_{attribute}_allocations']
    prob = solver_result['prob']
    nakamoto_coefficients = nakamoto_result[f'nakamoto_coefficients_{attribute}']
    
    list_current = list(node_df_current[attribute].unique())
    # Ensure the DataFrame is not empty before extracting unique values
    if not node_df_synthetic.empty:
        list_synthetic = list(node_df_synthetic[attribute].unique())
    else:
        list_synthetic = []

    
    # Pivot the DataFrame
    pivot_df = df_allocations.pivot(index='subnet_index', columns=f'{attribute}', values='allocation_value').fillna(0)

    # Create custom color mapping
    current_colors = plt.cm.Blues(np.linspace(0.3, 1, len(list_current)))
    synthetic_colors = plt.cm.Reds(np.linspace(0.3, 1, len(list_synthetic)))

    current_color_map = dict(zip(list_current, current_colors))
    synthetic_color_map = dict(zip(list_synthetic, synthetic_colors))
    color_map = {**current_color_map, **synthetic_color_map}

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=(15, 10))
    bottoms = np.zeros(len(pivot_df))

    for attr in attribute_list:
        values = pivot_df.get(attr, 0)
        ax.bar(pivot_df.index, values, bottom=bottoms, label=attr, 
               color=color_map.get(attr, 'black'),
               edgecolor='grey', 
               linewidth=1.5)  
        bottoms += values

    # Add Nakamoto coefficients as text above each bar
    for i, subnet_idx in enumerate(pivot_df.index):
        realized_nakamoto = nakamoto_coefficients[subnet_idx]
        ax.text(i, bottoms[i] + 2, f'{realized_nakamoto}', ha='center')

    # Add labels and title
    ax.set_xlabel('Subnet Index')
    ax.set_ylabel('Number of Nodes')
    ax.set_title(f'Node Allocations Per Subnet by {attribute.capitalize()}')
    num_legend_items = len(attribute_list)
    columns = max( 1, round( num_legend_items/35 ) )
    ax.legend(title=f'{attribute.capitalize()}s', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=columns)


    objective_value = pulp.value(prob.objective)
    total_nakamoto = sum(nakamoto_coefficients.values())
    
    ax.text(0.65, 0.95, f"Model: {prob.objective_name}", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.65, 0.92, f"Wall Clock Time: {elapsed_time:.2f} sec", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.65, 0.89, f"Objective Value: {objective_value:.2f}", transform=ax.transAxes, verticalalignment='top')
    ax.text(0.65, 0.86, f"Total Nakamoto Coefficients: {total_nakamoto:.2f}", transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()
    # Specify output folder
    output_folder = 'output'
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot into the output folder
    plt.savefig(os.path.join(output_folder, f"{attribute}_node_allocation.png"))
    
    plt.show(block=False)
    plt.pause(1)  
    plt.close()
    



def visualize_node_topology_matrix_with_double_rows_per_country(network_data):
    # 1. Filter data for is_synthetic = False
    df = network_data['node_df'][network_data['node_df']['is_synthetic'] == False]
    
    # 2. Group and sum data by country
    country_counts = df['country'].value_counts().sort_values(ascending=False)
    
    # 3. Create a color map for the countries
    country_current = country_counts.index.tolist()
  
    # Generate a set of unique colors for up to 40 dimension values using a combination of colormaps
    color_set_1 = plt.cm.tab20(np.arange(20))
    color_set_2 = plt.cm.tab20c(np.arange(20))
    current_colors = [color_set_1[i%20] if i%2 == 0 else color_set_2[i%20] for i in range(len(country_current))]
    color_map = dict(zip(country_current, current_colors))
    
    # Determine matrix width (limit to 60) and height
    matrix_width = min(country_counts.iloc[0] // 2, 60)
    max_subnet_size = max(network_data['network_topology']['subnet_size'])
    matrix_height = max(max_subnet_size, len(country_current) * 2)
    
    fig, ax = plt.subplots(figsize=(matrix_width/4, matrix_height/4))
    
    # Plot country nodes
    y_pos_counter = [0] * matrix_width  # Keeps track of the y-position for each column
    for country, count in country_counts.items():
        actual_count = count  # Store the original count for display
        if count > 100:
            count = 100
        
        for j in range(count):
            ypos = y_pos_counter[j // 2] * 2 + j % 2
            xpos = j // 2
            ax.add_patch(plt.Rectangle((xpos, ypos), 0.8, 0.8, color=color_map[country], linewidth=1))
            y_pos_counter[j // 2] += 1 if j % 2 == 1 else 0
        
        # Add the count text above the tiles
        if actual_count > 100:
            y_position_for_label = y_pos_counter[(count-1) // 2] * 2 -1
            ax.text(matrix_width + 1, y_position_for_label, str(actual_count), va='center')

    # Plot subnets
    for i, subnet_size in enumerate(network_data['network_topology']['subnet_size']):
        # Overlay cross on tiles up to the subnet size
        for j in range(subnet_size):
            ypos = j
            xpos = i
            cross_length = 0.2  # adjust to preference
            ax.plot([xpos + 0.4 - cross_length, xpos + 0.4 + cross_length], [ypos + 0.4 - cross_length, ypos + 0.4 + cross_length], color='black')
            ax.plot([xpos + 0.4 - cross_length, xpos + 0.4 + cross_length], [ypos + 0.4 + cross_length, ypos + 0.4 - cross_length], color='black')
            
    ax.set_xlim(0, matrix_width)
    ax.set_ylim(0, matrix_height)
    ax.set_xlabel('Number of Nodes')
    ax.set_ylabel('Country')    
    ax.set_title('Node topololgy matrix by country')
    
    # Add the legend for countries
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[country]) for country in country_current]
    ax.legend(handles, country_current, title='Countries', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Create a custom legend entry for the subnet crosses
    cross_legend = mlines.Line2D([0], [0], color='black', lw=0, marker='x', markersize=10, label='Subnet slots')
    
    # Add the custom legend entry to the list of handles and labels
    handles.append(cross_legend)
    country_current.append('Subnet slot')
    
    # Add the legend for countries including the subnet crosses
    ax.legend(handles, country_current, title='Countries', bbox_to_anchor=(1.05, 1), loc='upper left')
   
    
    plt.tight_layout()
    # Specify output folder
    output_folder = 'output'
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot into the output folder
    plt.savefig(os.path.join(output_folder, "country_double_row_topology_matrix.png"))
    
    plt.show(block=False)
    plt.pause(1)  
    plt.close()


def visualize_current_subnets(network_data):
  
    # Determine matrix width and height
    matrix_width = max(network_data['network_topology']['subnet_size'])
    matrix_height = len(network_data['network_topology']['subnet_size'])
    
    fig, ax = plt.subplots(figsize=(matrix_width/4, matrix_height/4))
    
    # Plot subnets
    for i, subnet_size in enumerate(network_data['network_topology']['subnet_size']):
        for j in range(subnet_size):
            ypos = j
            xpos = i
            cross_length = 0.2  # adjust to preference
            ax.plot([xpos + 0.4 - cross_length, xpos + 0.4 + cross_length], [ypos + 0.4 - cross_length, ypos + 0.4 + cross_length], color='black')
            ax.plot([xpos + 0.4 - cross_length, xpos + 0.4 + cross_length], [ypos + 0.4 + cross_length, ypos + 0.4 - cross_length], color='black')
    
    ax.set_xlim(0, matrix_height)  # matrix_height here as we are going by the number of subnets
    ax.set_ylim(0, matrix_width)  # matrix_width here as it represents the maximum number of nodes in a subnet
    ax.set_xlabel('Subnets')
    ax.set_ylabel('Number of nodes')
    ax.set_title('Slots of current subnets')
    
    # Legend
    cross_legend = mlines.Line2D([0], [0], color='black', lw=0, marker='x', markersize=10, label='Subnet slot')
    ax.legend(handles=[cross_legend], loc='upper right')
    
    plt.tight_layout()
    plt.show()


def visualize_node_topology_matrix(network_data, dimension, display_subnet_slots=True):
    # Convert dimension to a display-friendly format
    displayed_dimension = " ".join(dimension.split("_")).title()

    # 1. Filter data for is_synthetic = False
    df = network_data['node_df'][network_data['node_df']['is_synthetic'] == False]
    
    # 2. Group and sum data by the given dimension
    dimension_counts = df[dimension].value_counts().sort_values(ascending=False)
    
    # 3. Create a color map for the dimension values
    current_dimensions = dimension_counts.index.tolist()
    
    # Generate a set of unique colors for up to 40 dimension values using a combination of colormaps
    color_set_1 = plt.cm.tab20(np.arange(20))
    color_set_2 = plt.cm.tab20c(np.arange(20))
    current_colors = [color_set_1[i%20] if i%2 == 0 else color_set_2[i%20] for i in range(len(current_dimensions))]
    color_map = dict(zip(current_dimensions, current_colors))

    max_width = 60
    matrix_width = min(dimension_counts.iloc[0], max_width)
    max_subnet_size = max(network_data['network_topology']['subnet_size'])
    matrix_height = max(max_subnet_size, len(current_dimensions))
    
    fig, ax = plt.subplots(figsize=(matrix_width/4, matrix_height/4))
    
    # Plot nodes for each dimension value
    y_pos_counter = [0] * matrix_width
    for dim_val, count in dimension_counts.items():
        actual_count = count
        if count > max_width:
            count = max_width
        
        for j in range(count):
            ypos = y_pos_counter[j]
            xpos = j
            ax.add_patch(plt.Rectangle((xpos, ypos), 0.7, 0.7, color=color_map[dim_val],  linewidth=1))
            y_pos_counter[j] += 1
        
        if actual_count > max_width:
            y_position_for_label = y_pos_counter[count-1]-1
            ax.text(matrix_width + 1, y_position_for_label, str(actual_count))

    # Plot subnets
    if display_subnet_slots:
        for i, subnet_size in enumerate(network_data['network_topology']['subnet_size']):
            for j in range(subnet_size):
                ypos = j
                xpos = i
                cross_length = 0.2  # adjust to preference
                ax.plot([xpos + 0.4 - cross_length, xpos + 0.4 + cross_length], [ypos + 0.4 - cross_length, ypos + 0.4 + cross_length], color='black')
                ax.plot([xpos + 0.4 - cross_length, xpos + 0.4 + cross_length], [ypos + 0.4 + cross_length, ypos + 0.4 - cross_length], color='black')
    
    ax.set_xlim(0, matrix_width)
    ax.set_ylim(0, matrix_height)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel(displayed_dimension)
    ax.set_title(f'Node topology matrix by {displayed_dimension}')
    
    # Add the legend for dimension values and the cross legend for subnets
    handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[dim_val]) for dim_val in current_dimensions]
    if display_subnet_slots:
        cross_legend = mlines.Line2D([0], [0], color='black', lw=0, marker='x', markersize=10, label='Subnet slot')
        handles.append(cross_legend)
        current_dimensions.append('Subnet slot')
      
    ax.legend(handles, current_dimensions, title=displayed_dimension, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Specify output folder
    output_folder = 'output'
    
    # Check if the folder exists, if not, create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the plot into the output folder
    plt.savefig(os.path.join(output_folder, f"{dimension}_topology_matrix.png"))
    
    plt.show(block=False)
    plt.pause(1)  
    plt.close()
    





