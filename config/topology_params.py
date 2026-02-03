"""
Topology configuration parameters for fog computing simulation.
Based on YAFS 3.1 standards.
"""
import random
import networkx as nx

# CLOUD NODE ATTRIBUTES (experimentConfiguration: CLOUDCAPACITY, CLOUDSTORAGE, CLOUDSPEED)
CLOUD_ATTRS = {
    "IPT": 10000,  # INSTR per MS (CLOUDSPEED)
    "RAM": 9999999999999999,  # MB (CLOUDCAPACITY)
    "TB": 99999,  # TB Storage (CLOUDSTORAGE)
    "type": "CLOUD",
    "model": "cloud",
    "WATT": 500.0,  # Power (e.g. watts); energy = time_service * WATT in YAFS
}

# CLOUD GATEWAY (CFG) LINK ATTRIBUTES (CLOUDBW, CLOUDPR)
CLOUD_LINK_ATTRS = {
    "BW": 125000,  # BYTES / MS --> 1000 Mbits/s (CLOUDBW)
    "PR": 1,  # MS (CLOUDPR)
}

# NETWORK GENERATION (func_NETWORKGENERATION: barabasi_albert n=100, m=2)
NETWORK_CONFIG = {
    "generator": "barabasi_albert_graph",
    "params": {"n": 100, "m": 2},
}

# FOG GATEWAY DISTRIBUTION (experimentConfiguration)
PERCENTAGE_OF_GATEWAYS = 0.25  # FG - Fog Gateways (PERCENTATGEOFGATEWAYS)
PERCENTAGE_OF_CLOUD_GATEWAYS = 0.05  # CFG - Cloud-Fog Gateways (PERCENTAGEOFCLOUDGATEWAYS)

# FOG NODE ATTRIBUTES (func_NODERESOURECES, func_NODESPEED, func_NODESTORAGE)
def get_fog_node_attrs(node_id):
    """Generate random attributes for fog nodes. Type (FOG/FG/CFG) set in generator."""
    return {
        "id": node_id,
        "IPT": random.randint(100, 1000),  # INTS / MS (func_NODESPEED)
        "RAM": random.randint(10, 25),  # MB RAM (func_NODERESOURECES)
        "TB": random.uniform(1, 10),  # TB (func_NODESTORAGE)
        "model": "fog",
        "WATT": random.uniform(5.0, 50.0),  # Power (watts); for energy consumption metric
    }

# FOG LINK ATTRIBUTES (func_PROPAGATIONTIME, func_BANDWITDH)
def get_fog_link_attrs():
    """Generate random attributes for fog links."""
    return {
        "BW": random.randint(75000, 75000),  # BYTES / MS (func_BANDWITDH)
        "PR": random.randint(5, 5),  # MS (func_PROPAGATIONTIME)
    }

# TOPOLOGY OUTPUT
OUTPUT_FILE = "scenarios/networkDefinition.json"
