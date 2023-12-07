"""
Import topology from the Public Dashboard

Swagger definition of the API is at: https://ic-api.internetcomputer.org/api/v3/swagger
"""

from typing import Any, Dict
import requests
import pandas
from pathlib import Path
from datetime import datetime

BASE_URL = "https://ic-api.internetcomputer.org/api/v3"

# Get a list of subnets from /api/v3/subnets
def get_subnets() -> Dict[str, Dict[str, Any]]:
    url = BASE_URL + "/subnets"
    response = requests.get(url)
    response_json = response.json()
    result = []
    for subnet in response_json["subnets"]:
        del subnet["replica_versions"]
        result.append(subnet)
    return result


# Get a list of nodes from /api/v3/nodes
def get_nodes():
    url = BASE_URL + "/nodes"
    response = requests.get(url)
    response_json = response.json()
    return response_json["nodes"]


def main():
    print("IC Subnets:")
    df_subnets = pandas.DataFrame.from_records(get_subnets()).set_index("subnet_id")
    print(df_subnets)

    print("\nIC Nodes:")
    df_nodes = pandas.DataFrame.from_records(get_nodes()).set_index("node_id")
    print(df_nodes)
    
    # Generate a filename with the current date and time
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"current_nodes_{current_time}.csv"

    # Define the path to the ../data directory relative to this script
    script_dir = Path(__file__).resolve().parent  # Get the directory where the script is located
    data_dir = script_dir.parent / 'data'  # Path to the ../data directory

    # Check if the data directory exists, create if it doesn't
    data_dir.mkdir(parents=True, exist_ok=True)

    # Full path for the file
    file_path = data_dir / filename

    # Save the DataFrame to a CSV file in the data directory
    df_nodes.to_csv(file_path, index=True)

    print(f"Saved current nodes to {file_path}")


if __name__ == "__main__":
    main()