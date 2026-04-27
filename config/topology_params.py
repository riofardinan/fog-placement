"""
Topology configuration parameters for fog computing simulation.
Based on Pakpahan et al. (2025) [1] and YAFS 3.1 standards [2, 3].
"""
import random
import networkx as nx

# Number of Nodes
NUM_NODES = 100
# Propagation Time (ms)
PROPAGATION_TIME_MIN = 10
PROPAGATION_TIME_MAX = 10
# Bandwidth (bytes/ms)
BANDWIDTH_MIN = 75000
BANDWIDTH_MAX = 75000

# RAM (MB RAM)
MIN_RAM = 10
MAX_RAM = 25
# IPT (Instructions/ms, IPT)
MIN_IPT = 100
MAX_IPT = 1000
# Storage (TB)
# STORAGE_MIN = 0.2
# STORAGE_MAX = 100.0

# CFG (Cloud-Fog Gateway): 25% node dengan centrality TERTINGGI → cloud terhubung ke sini [1]
PERCENTAGE_OF_CLOUD_GATEWAYS = 0.25
# FG (Fog Gateway/Edge): 25% node dengan centrality TERENDAH → 25 gateway nodes [1]
PERCENTAGE_OF_GATEWAYS = 0.25

# CLOUD NODE ATTRIBUTES
CLOUD_ATTRS = {
    "IPT": 10000,
    "RAM": 99999,
    # "TB": 99999,
    "type": "CLOUD",
    # "WATT": 500.0,
}

# CLOUD GATEWAY (CFG) LINK ATTRIBUTES
CLOUD_LINK_ATTRS = {
    "BW": 75000,
    "PR": 10,
}

def get_fog_node_attrs(node_id):
    """Generate random attributes for fog nodes (min/max resources, speed, storage)."""
    return {
        "id": node_id,
        "IPT": random.randint(MIN_IPT, MAX_IPT),
        "RAM": random.randint(MIN_RAM, MAX_RAM),
        # "TB": random.uniform(STORAGE_MIN, STORAGE_MAX),
        # "model": "fog",
        # "WATT": random.uniform(5.0, 50.0),
    }

def get_fog_link_attrs():
    """Generate fog link attributes (propagation_time and bandwidth in range)."""
    return {
        "PR": random.randint(PROPAGATION_TIME_MIN, PROPAGATION_TIME_MAX),
        "BW": random.randint(BANDWIDTH_MIN, BANDWIDTH_MAX),
    }

# TOPOLOGY OUTPUT
OUTPUT_FILE = "scenarios/networkDefinition.json"
