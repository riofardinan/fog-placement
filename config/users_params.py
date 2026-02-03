"""
Users/IoT devices configuration parameters for fog computing simulation.
Based on YAFS 3.1 standards.
"""
import random

# USER REQUEST GENERATION
def get_request_probability():
    """
    App popularity threshold.
    Determines the probability that a device has requests associated with an app.
    """
    return random.random() / 4  # func_REQUESTPROB

def get_user_request_rate():
    """
    User request rate (inter-arrival time).
    """
    return random.randint(200, 1000)  # MS (func_USERREQRAT)

# USERS OUTPUT
OUTPUT_FILE = "scenarios/usersDefinition.json"
