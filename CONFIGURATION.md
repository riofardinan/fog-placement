# Configuration Guide

Panduan lengkap untuk mengkonfigurasi simulasi fog computing.

## Overview

Konfigurasi dibagi menjadi 3 file:

1. `config/topology_params.py` - Cloud, fog nodes, network topology
2. `config/app_params.py` - Applications, services, messages
3. `config/users_params.py` - User sources, request patterns

## 1. Topology Configuration (`topology_params.py`)

### Cloud Node

```python
CLOUD_ATTRS = {
    "model": "cloud",           # Tag untuk identifikasi
    "IPT": 10000,               # Instructions Per Time (INSTR/MS)
    "RAM": 9999999999999999,    # Memory (MB)
    "STORAGE": 99999,           # Storage (TB)
    "COST": 10,                 # Cost per time unit
    "WATT": 1000.0             # Power consumption (Watts)
}
```

**Parameter mapping dari kriteria:**

- `IPT` = CLOUDSPEED (10000)
- `RAM` = CLOUDCAPACITY (unlimited)
- `STORAGE` = CLOUDSTORAGE (99999 TB)

### Cloud-Fog Gateway Links

```python
CLOUD_LINK_ATTRS = {
    "BW": 125000,  # Bandwidth: BYTES/MS → 1000 Mbits/s
    "PR": 1        # Propagation delay: MS
}
```

**Parameter mapping:**

- `BW` = CLOUDBW (125000 BYTES/MS)
- `PR` = CLOUDPR (1 MS)

### Network Generation

```python
NETWORK_CONFIG = {
    "generator": "barabasi_albert_graph",
    "params": {
        "n": 100,  # Total fog nodes
        "m": 2     # Edges to attach from new node
    }
}
```

**Available NetworkX generators:**

- `barabasi_albert_graph(n, m)` - Scale-free network (default)
- `erdos_renyi_graph(n, p)` - Random graph
- `watts_strogatz_graph(n, k, p)` - Small-world network
- `powerlaw_cluster_graph(n, m, p)` - Power-law cluster
- `random_regular_graph(d, n)` - Regular graph

### Gateway Distribution

```python
PERCENTAGE_OF_GATEWAYS = 0.25        # 25% nodes as Fog Gateways (FG)
PERCENTAGE_OF_CLOUD_GATEWAYS = 0.05  # 5% nodes as Cloud-Fog Gateways (CFG)
```

**Gateway types:**

- **CFG (Cloud-Fog Gateway):** Connected to cloud, high-capacity
- **FG (Fog Gateway):** Edge gateways, intermediate capacity
- **Regular:** Normal fog nodes

### Fog Node Attributes (Random)

```python
def get_fog_node_attrs(node_id):
    return {
        "id": node_id,
        "model": "fog",
        "IPT": random.randint(100, 1000),    # Speed varies
        "RAM": random.randint(10, 25),       # Memory varies
        "STORAGE": random.uniform(1, 10),    # Storage varies
        "COST": 1,
        "WATT": 50.0
    }
```

**Parameter mapping:**

- `IPT` = func_NODESPEED (100-1000)
- `RAM` = func_NODERESOURECES (10-25 MB)
- `STORAGE` = func_NODESTORAGE (1-10 TB)

### Fog Link Attributes (Random)

```python
def get_fog_link_attrs():
    return {
        "BW": random.randint(75000, 75000),  # Bandwidth
        "PR": random.randint(5, 5)           # Propagation
    }
```

**Parameter mapping:**

- `BW` = func_BANDWITDH (75000 BYTES/MS)
- `PR` = func_PROPAGATIONTIME (5 MS)

## 2. Application Configuration (`app_params.py`)

### Number of Applications

```python
TOTAL_NUMBER_OF_APPS = 20  # Total applications to generate
```

### Application DAG Generation

```python
def generate_app_dag():
    """Generate DAG with 2-10 services per application."""
    num_services = random.randint(2, 10)
    return nx.gn_graph(num_services)  # Growing Network graph
```

**Alternative DAG generators:**

- `nx.gn_graph(n)` - Growing network (default)
- `nx.gnr_graph(n, p)` - Growing network with redirection
- `nx.gnc_graph(n)` - Growing network with copying
- Custom DAG: `nx.DiGraph()` with manual edges

### Service Attributes

```python
def get_service_attrs():
    return {
        "instructions": random.randint(20000, 60000),    # Processing required
        "bytes": random.randint(1500000, 4500000),       # Message size
        "RAM": random.randint(1, 6)                      # Memory required
    }
```

**Parameter mapping:**

- `instructions` = func_SERVICEINSTR (20000-60000)
  - With NODESPEED (100-1000), gives ~20-600 MS processing time
- `bytes` = func_SERVICEMESSAGESIZE (1500000-4500000)
  - With BANDWITDH (75000), gives ~20-60 MS transmission time
- `RAM` = func_SERVICERESOURCES (1-6 MB)

### Application Deadline

```python
def get_app_deadline():
    return random.randint(2600, 6600)  # MS
```

**Parameter mapping:**

- Deadline = func_APPDEADLINE (2600-6600 MS)

## 3. Users Configuration (`users_params.py`)

### Request Probability (App Popularity)

```python
def get_request_probability():
    """Probability that a device requests this app."""
    return random.random() / 4  # 0-0.25
```

**Parameter mapping:**

- func_REQUESTPROB: `random.random()/4`
- Higher value = more popular app

### Request Rate (Inter-arrival Time)

```python
def get_user_request_rate():
    """Inter-arrival time between requests."""
    return random.randint(200, 1000)  # MS
```

**Parameter mapping:**

- func_USERREQRAT (200-1000 MS)
- Lower value = higher request frequency

## Example Scenarios

### Scenario 1: High-Density Fog

**Dense fog network with many nodes:**

```python
# topology_params.py
NETWORK_CONFIG["params"]["n"] = 200  # 200 fog nodes
NETWORK_CONFIG["params"]["m"] = 4    # More connections
PERCENTAGE_OF_GATEWAYS = 0.3         # More gateways
```

### Scenario 2: Cloud-Heavy

**Most processing on cloud:**

```python
# topology_params.py
PERCENTAGE_OF_CLOUD_GATEWAYS = 0.2  # Many CFG nodes

# app_params.py
def get_service_attrs():
    return {
        "instructions": random.randint(50000, 100000),  # Heavy services
        "bytes": random.randint(5000000, 10000000),     # Large messages
        "RAM": random.randint(10, 20)                   # High memory
    }
```

### Scenario 3: Edge-Heavy

**Most processing at edge:**

```python
# topology_params.py
def get_fog_node_attrs(node_id):
    return {
        "IPT": random.randint(500, 2000),   # Powerful fog nodes
        "RAM": random.randint(20, 50),      # More memory
        # ...
    }

# app_params.py
TOTAL_NUMBER_OF_APPS = 50  # Many small apps
def generate_app_dag():
    return nx.gn_graph(random.randint(2, 5))  # Smaller apps
```

### Scenario 4: High Workload

**Heavy user load:**

```python
# users_params.py
def get_request_probability():
    return random.random() / 2  # More users per app (0-0.5)

def get_user_request_rate():
    return random.randint(50, 200)  # Higher frequency
```

## Advanced: Custom Distributions

### Custom Network Topology

```python
# topology_params.py
import networkx as nx

def custom_topology():
    """Create custom hierarchical topology."""
    G = nx.Graph()
    
    # Layer 1: Core (connected to cloud)
    core_nodes = range(1, 6)  # 5 core nodes
    G.add_nodes_from(core_nodes)
    G.add_edges_from([(i, j) for i in core_nodes for j in core_nodes if i < j])
    
    # Layer 2: Edge (connected to core)
    edge_nodes = range(6, 56)  # 50 edge nodes
    for edge in edge_nodes:
        core = random.choice(list(core_nodes))
        G.add_edge(edge, core)
    
    return G

# Use in generate_scenario.py:
# fog_graph = custom_topology()
```

### Custom Application Pattern

```python
# app_params.py
def create_pipeline_app():
    """Create pipeline application (linear DAG)."""
    num_stages = random.randint(3, 7)
    dag = nx.DiGraph()
    dag.add_nodes_from(range(num_stages))
    dag.add_edges_from([(i, i+1) for i in range(num_stages-1)])
    return dag
```

### Custom User Distribution

```python
# users_params.py
def get_hotspot_users(topology, num_hotspots=5):
    """Create user hotspots (concentrated users on few nodes)."""
    fog_nodes = [n["id"] for n in topology["entity"] if n["id"] != 0]
    hotspots = random.sample(fog_nodes, num_hotspots)
    
    # 80% of users on hotspot nodes
    return hotspots
```

