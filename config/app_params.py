"""
Application configuration parameters for fog computing simulation.
Sequential (linear chain) microservice DAG — Pakpahan et al. (2025) [1].
"""
import random
import networkx as nx

# Number of Applications
NUM_APPLICATIONS = 5
# Services Per Application
MODULES_PER_APP_MIN = 2
MODULES_PER_APP_MAX = 8
# RAM (MB RAM)
RAM_USAGE_MIN = 1
RAM_USAGE_MAX = 6

# Instructions Per Request
INSTRUCTIONS_PER_REQ_MIN = 20000
INSTRUCTIONS_PER_REQ_MAX = 60000
# Message Size (bytes)
MESSAGE_SIZE_MIN = 1500000
MESSAGE_SIZE_MAX = 4500000
# Deadline (ms)
DEADLINE_MIN = 300
DEADLINE_MAX = 50000

def generate_app_dag():
    """Generate sequential linear chain DAG (microservice chain) [1]."""
    num_modules = random.randint(MODULES_PER_APP_MIN, MODULES_PER_APP_MAX)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_modules))
    dag.add_edges_from((i, i + 1) for i in range(num_modules - 1))
    return dag

def get_service_attrs():
    """Generate random service attributes (instructions, message size, RAM) — uniform."""
    return {
        "instructions": random.randint(INSTRUCTIONS_PER_REQ_MIN, INSTRUCTIONS_PER_REQ_MAX),
        "bytes": random.randint(MESSAGE_SIZE_MIN, MESSAGE_SIZE_MAX),
        "RAM": random.randint(RAM_USAGE_MIN, RAM_USAGE_MAX),
    }

def get_app_deadline():
    """Generate random deadline (uniform deadline_min–deadline_max ms)."""
    return random.randint(DEADLINE_MIN, DEADLINE_MAX)

# APPLICATION OUTPUT
OUTPUT_FILE = "scenarios/appDefinition.json"
