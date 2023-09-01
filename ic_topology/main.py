"""
Import topology from the Public Dashboard

Swagger definition of the API is at: https://ic-api.internetcomputer.org/api/v3/swagger
"""

from typing import Any, Dict
import requests
import pandas

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


if __name__ == "__main__":
    main()