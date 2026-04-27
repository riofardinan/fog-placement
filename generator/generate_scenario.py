"""
Scenario Generator for YAFS 3.1
Generates topology, applications, and users configuration as JSON files.

This is RUN 1: Generate all configurations before simulation.
"""
import json
import random
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import operator
import networkx as nx
from config import topology_params as topo_cfg
from config import app_params as app_cfg
from config import users_params as user_cfg


def generate_topology(seed=42):
    print("Generating topology...")

    random.seed(seed)
    fog_graph = nx.barabasi_albert_graph(n=topo_cfg.NUM_NODES, m=2, seed=seed)
    
    num_fog_nodes = fog_graph.number_of_nodes()
    cloud_id = num_fog_nodes  # 100
    
    # Gateway selection by betweenness centrality
    centrality_no_order = nx.betweenness_centrality(fog_graph, weight="weight")
    centrality_sorted = sorted(
        centrality_no_order.items(),
        key=operator.itemgetter(1),
        reverse=True
    )
    
    num_cfg = max(1, int(num_fog_nodes * topo_cfg.PERCENTAGE_OF_CLOUD_GATEWAYS))
    num_fg = int(num_fog_nodes * topo_cfg.PERCENTAGE_OF_GATEWAYS)
    
    # CFG = top nodes by centrality (highest)
    cloud_fog_gateways = {centrality_sorted[i][0] for i in range(num_cfg)}
    # FG = bottom 25% (lowest centrality)
    fog_gateways = {
        centrality_sorted[id_dev][0]
        for id_dev in range(len(centrality_sorted) - num_fg, len(centrality_sorted))
        if id_dev >= 0
    }
    
    # Build topology
    topology = {"entity": [], "link": []}
    
    # Fog nodes (id 0..99) with type FOG | FG | CFG
    for node_id in fog_graph.nodes():
        fog_node = topo_cfg.get_fog_node_attrs(node_id)
        if node_id in cloud_fog_gateways:
            fog_node["type"] = "CFG"
        elif node_id in fog_gateways:
            fog_node["type"] = "FG"
        else:
            fog_node["type"] = "FOG"
        topology["entity"].append(fog_node)
    
    # Cloud node (id=100)
    cloud_node = topo_cfg.CLOUD_ATTRS.copy()
    cloud_node["id"] = cloud_id
    topology["entity"].append(cloud_node)
    
    # Fog-to-fog links (s, d in 0..99)
    for edge in fog_graph.edges():
        link = topo_cfg.get_fog_link_attrs()
        link["s"] = edge[0]
        link["d"] = edge[1]
        topology["link"].append(link)
    
    # CFG-to-cloud links (experimentConfiguration: cloudGtw -> cloudId)
    for cfg_node in cloud_fog_gateways:
        link = topo_cfg.CLOUD_LINK_ATTRS.copy()
        link["s"] = cfg_node
        link["d"] = cloud_id
        topology["link"].append(link)
    
    print(f"  - Created {len(topology['entity'])} nodes (fog 0..{num_fog_nodes - 1}, cloud {cloud_id})")
    print(f"  - Created {len(topology['link'])} links")
    print(f"  - Gateways (centrality): {num_cfg} CFG, {num_fg} FG")
    
    return topology


def generate_applications(seed=42, num_apps=None):
    random.seed(seed)

    total = num_apps if num_apps is not None else app_cfg.NUM_APPLICATIONS
    print(f"Generating {total} applications...")
    
    applications = []

    for app_id in range(total):
        # Generate DAG (experimentConfiguration: func_APPGENERATION)
        dag = app_cfg.generate_app_dag()
        
        # Reverse edges (experimentConfiguration)
        edge_list = list(dag.edges())
        dag.remove_edges_from(edge_list)
        dag.add_edges_from((v, u) for u, v in edge_list)
        
        # Source = first in topological order after reverse
        topo_order = list(nx.topological_sort(dag))
        source_node = topo_order[0]
        
        deadline = app_cfg.get_app_deadline()
        
        # App structure (YAFS + experimentConfiguration: id, name, deadline, MaxLatency)
        app = {
            "id": app_id,
            "name": str(app_id),
            "deadline": deadline,
            "HwReqs": 1,
            "MaxReqs": 200,
            "MaxLatency": deadline,
            "transmission": [],
            "module": [],
            "message": []
        }
        
        i = app_id
        module_name = lambda n: str(i) + "_" + str(n)
        
        # Per-node RAM (experimentConfiguration: servicesResources)
        services_resources = {}
        for n in dag.nodes():
            services_resources[n] = app_cfg.get_service_attrs()["RAM"]
        
        # 1) Modules (experimentConfiguration: myNode id, name, RAM, type)
        for n in dag.nodes():
            app["module"].append({
                "id": n,
                "name": module_name(n),
                "RAM": services_resources[n],
                "type": "MODULE"
            })
        
        edge_number = 0
        
        # 2) Source message: one per app "M.USER.APP.i" (experimentConfiguration)
        src_attrs = app_cfg.get_service_attrs()
        app["message"].append({
            "id": edge_number,
            "name": "M.USER.APP." + str(i),
            "s": "None",
            "d": module_name(source_node),
            "instructions": src_attrs["instructions"],
            "bytes": src_attrs["bytes"]
        })
        edge_number += 1
        
        # Transmissions for source node: message_in M.USER.APP.i, message_out per outgoing edge
        for o in dag.edges():
            if o[0] == source_node:
                app["transmission"].append({
                    "module": module_name(source_node),
                    "message_in": "M.USER.APP." + str(i),
                    "message_out": str(i) + "_(" + str(o[0]) + "-" + str(o[1]) + ")"
                })
        
        # 3) Edge messages: name "i_(u-v)" (experimentConfiguration)
        for n in dag.edges():
            u, v = n[0], n[1]
            edge_attrs = app_cfg.get_service_attrs()
            app["message"].append({
                "id": edge_number,
                "name": str(i) + "_(" + str(u) + "-" + str(v) + ")",
                "s": module_name(u),
                "d": module_name(v),
                "instructions": edge_attrs["instructions"],
                "bytes": edge_attrs["bytes"]
            })
            edge_number += 1
            
            # Transmissions for module at v: message_in from this edge, message_out per successor edge
            dest_node = v
            for o in dag.edges():
                if o[0] == dest_node:
                    app["transmission"].append({
                        "module": module_name(n[1]),
                        "message_in": str(i) + "_(" + str(u) + "-" + str(v) + ")",
                        "message_out": str(i) + "_(" + str(o[0]) + "-" + str(o[1]) + ")"
                    })
        
        # 4) Sink nodes: no outgoing edges; one transmission with message_in only (experimentConfiguration)
        for n in dag.nodes():
            if dag.out_degree(n) == 0:
                for m in dag.edges():
                    if m[1] == n:
                        app["transmission"].append({
                            "module": module_name(n),
                            "message_in": str(i) + "_(" + str(m[0]) + "-" + str(m[1]) + ")"
                        })
                        break
        
        applications.append(app)
    
    print(f"  - Created {len(applications)} applications")
    
    return applications


def generate_users(topology, applications, seed=42):
    random.seed(seed)
    
    print(f"Generating users/sources (1 per app)...")
    
    users = {"sources": []}
    
    # Only FG (fog gateway) nodes host users
    gateway_nodes = [
        entity["id"] for entity in topology["entity"]
        if entity.get("type") == "FG"
    ]
    
    if not gateway_nodes:
        gateway_nodes = [
            entity["id"] for entity in topology["entity"]
            if entity.get("type") != "CLOUD"
        ]
    
    # Build per-app: message name and app_name for sources
    app_info = []
    for app in applications:
        source_messages = [msg for msg in app["message"] if msg["s"] == "None"]
        if not source_messages:
            continue
        app_info.append({
            "app_id": app["id"],
            "app_name": str(app["id"]),
            "message_name": source_messages[0]["name"],
        })
    
    if not app_info:
        print("  - No apps with source messages, skipping users")
        return users
    
    # Per-app lambda [200–1000] ms — paper Table 1: "IoT request rate 200–1000" [1]
    # seed % 100 keeps lambda distribution consistent across different app-count scenarios
    random.seed(seed % 100)

    # Exactly 1 source per app — round-robin across gateways for even load distribution [1]
    for idx, info in enumerate(app_info):
        node_id = gateway_nodes[idx % len(gateway_nodes)]
        users["sources"].append({
            "id_resource": node_id,
            "app": info["app_name"],
            "message": info["message_name"],
            "lambda": user_cfg.get_user_request_rate(),
        })
    
    print(f"  - Created {len(users['sources'])} user sources (1 per app, FG gateways only)")
    
    return users


def main():
    # Create scenarios directory
    scenarios_dir = Path(__file__).parent.parent / "scenarios"
    scenarios_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    SEED = 42
    
    # Generate topology
    topology = generate_topology(seed=SEED)
    topo_file = scenarios_dir / "networkDefinition.json"
    with open(topo_file, 'w') as f:
        json.dump(topology, f, indent=2)
    print(f"Saved: {topo_file}")
    
    # Generate applications
    applications = generate_applications(seed=SEED)
    app_file = scenarios_dir / "appDefinition.json"
    with open(app_file, 'w') as f:
        json.dump(applications, f, indent=2)
    print(f"Saved: {app_file}")
    
    # Generate users
    users = generate_users(topology, applications, seed=SEED)
    user_file = scenarios_dir / "usersDefinition.json"
    with open(user_file, 'w') as f:
        json.dump(users, f, indent=2)
    print(f"Saved: {user_file}")
    
    print("\n" + "=" * 60)
    print("Generation completed!")
    print("=" * 60)
    print("\nNext step: Run generate_placements.py to create allocation files.")


if __name__ == "__main__":
    main()
