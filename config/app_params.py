"""
Application configuration parameters for fog computing simulation.
Based on YAFS 3.1 standards.
"""
import random
import networkx as nx

# APPLICATION GENERATION
TOTAL_NUMBER_OF_APPS = 20

# Application DAG generation function
def generate_app_dag():
    """
    Generate random DAG for application structure.
    Uses GN graph (growing network) with random number of nodes.
    """
    num_services = random.randint(2, 10)  # From func_APPGENERATION: nx.gn_graph(random.randint(2,10))
    return nx.gn_graph(num_services)

# SERVICE ATTRIBUTES (random distributions)
def get_service_attrs():
    """Generate random attributes for a service/module."""
    return {
        "instructions": random.randint(20000, 60000),  # INSTR (func_SERVICEINSTR)
        # Taking into account NODESPEED, this gives between 200-600 MS
        "bytes": random.randint(1500000, 4500000),  # BYTES (func_SERVICEMESSAGESIZE)
        # Taking account net bandwidth gives between 20-60 MS
        "RAM": random.randint(1, 6)  # MB of RAM (func_SERVICERESOURCES)
    }

# APPLICATION DEADLINE
def get_app_deadline():
    """Generate random deadline for application."""
    return random.randint(2600, 6600)  # MS (func_APPDEADLINE)

# APPLICATION OUTPUT
OUTPUT_FILE = "scenarios/appDefinition.json"
