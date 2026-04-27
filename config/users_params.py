"""
Users/IoT devices configuration parameters for fog computing simulation.
1 IoT source per application, placed at FG gateway nodes — Pakpahan et al. (2025) [1].
"""
import random

# request_interval (ms)
REQUEST_INTERVAL_MIN = 200
REQUEST_INTERVAL_MAX = 1000

def get_user_request_rate():
    """User request rate: inter-arrival time (uniform request_interval_min–max ms)."""
    return random.randint(REQUEST_INTERVAL_MIN, REQUEST_INTERVAL_MAX)

# USERS OUTPUT
OUTPUT_FILE = "scenarios/usersDefinition.json"
