"""
Complex Network based Placement (CNPlacement)
Based on Lera et al., IEEE IoT Journal 2019
Uses Girvan-Newman community detection + transitive closure partitioning.
"""
import itertools
import operator
import networkx as nx
from networkx.algorithms import community
from placements.placement import Placement

class CNPlacement(Placement):
    """
    CN-based placement using:
    - Girvan-Newman community detection
    - Transitive closure partition for app DAG
    - Device fitness (exec_time + network_time)
    - Sorted by app deadline (if available)
    """
    
    def __init__(self):
        super().__init__()
        self.name = "CNPlacement"
        self.sorted_communities = []
        self.node_busy_resources = {}
    
    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation using CN optimization.
        
        Args:
            topology: Topology dict with entities and links
            applications: List of application dicts
            users: Users dict with sources
        
        Returns:
            List of allocation dicts
        """
        # Build NetworkX graph with weights
        G = nx.Graph()
        entities = {e["id"]: e for e in topology["entity"]}
        
        for entity in topology["entity"]:
            G.add_node(entity["id"], **entity)
        
        for link in topology["link"]:
            # Weight = PR + msg_size/BW (use average message size)
            avg_msg_size = 2500000  # bytes (avg from config)
            weight = link["PR"] + avg_msg_size / link["BW"]
            G.add_edge(link["s"], link["d"], weight=weight, **link)
        
        # Find cloud node (YAFS: type CLOUD; or model cloud for backward compat)
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
                break
        
        # Initialize busy resources
        self.node_busy_resources = {n: 0.0 for n in G.nodes()}
        
        # Compute communities using Girvan-Newman
        self._compute_communities(G)
        
        # Build app graphs and compute transitive closures
        apps_closures = {}
        apps_graphs = {}
        for app in applications:
            app_graph = self._build_app_graph(app)
            apps_graphs[app["id"]] = app_graph
            if app_graph.number_of_nodes() > 0:
                source_nodes = [n for n in app_graph.nodes() if app_graph.in_degree(n) == 0]
                if source_nodes:
                    apps_closures[app["id"]] = self._transitive_closure_partition(
                        source_nodes[0], app_graph
                    )
        
        # Prepare allocation
        allocation = []
        
        # Sort apps by deadline (lower deadline first)
        sorted_apps = sorted(applications, key=lambda a: a.get("MaxLatency", 10000))
        
        # Get user gateways per app
        app_gateways = self._get_app_gateways(users)
        
        # Place each app
        for app in sorted_apps:
            app_id = app["id"]
            if app_id not in apps_closures:
                # No valid graph, place on cloud
                for module in app["module"]:
                    allocation.append({
                        "module_name": module["name"],
                        "app": str(app_id),
                        "id_resource": cloud_id
                    })
                continue
            
            gateways = app_gateways.get(app_id, [1])  # default gateway
            
            for gateway in gateways:
                # Try to place in communities (from deepest to shallowest)
                placed = False
                for comm, depth in self.sorted_communities:
                    if gateway in comm:
                        placement = self._place_app_in_community(
                            app, gateway, comm, apps_closures[app_id], entities, G
                        )
                        if placement:
                            for module_name, device_id in placement.items():
                                allocation.append({
                                    "module_name": module_name,
                                    "app": str(app_id),
                                    "id_resource": device_id
                                })
                            placed = True
                            break
                
                if not placed:
                    # Fallback: place on cloud
                    for module in app["module"]:
                        allocation.append({
                            "module_name": module["name"],
                            "app": str(app_id),
                            "id_resource": cloud_id
                        })
                break  # Only place once per app
        
        return allocation
    
    def _compute_communities(self, G):
        """Girvan-Newman community detection."""
        communities_gen = community.girvan_newman(G)
        communities = {frozenset(G.nodes()): 0}
        level = 1
        
        for comms in itertools.islice(communities_gen, G.number_of_nodes()):
            for c in comms:
                communities[frozenset(c)] = level
            level += 1
        
        self.sorted_communities = sorted(
            communities.items(), key=lambda x: x[1], reverse=True
        )
    
    def _transitive_closure_partition(self, source, app_graph):
        """Compute transitive closure partitions (Algorithm 2)."""
        closures = {}
        
        def dfs(node, level):
            closures.setdefault(level, set())
            desc = set(nx.descendants(app_graph, node)) | {node}
            fs = frozenset(desc)
            if fs not in closures[level]:
                closures[level].add(fs)
                for n in app_graph.neighbors(node):
                    closures.setdefault(level + 1, set()).add(frozenset([node]))
                    dfs(n, level + 1)
        
        dfs(source, 0)
        
        prev = closures[0]
        for lvl in sorted(closures.keys()):
            current = set().union(*closures[lvl])
            extra = {s for s in prev if len(s & current) == 0}
            closures[lvl] |= extra
            prev = closures[lvl]
        
        return closures
    
    def _place_app_in_community(self, app, gateway, community, closures, entities, G):
        """Place app services in a community using fitness ordering."""
        remaining = {m["name"] for m in app["module"]}
        placement = {}
        
        # Order devices by fitness
        ordered_devices = self._order_devices_by_fitness(community, gateway, app, entities, G)
        
        for dev in ordered_devices:
            if dev not in entities:
                continue
            free_res = entities[dev].get("RAM", 0) - self.node_busy_resources[dev]
            
            for lvl in sorted(closures.keys()):
                for sset in sorted(closures[lvl], key=len, reverse=True):
                    # Convert service IDs to module names
                    sset_names = {f"{app['id']}_{sid:02d}" for sid in sset}
                    if sset_names & remaining:
                        req = sum(m["RAM"] for m in app["module"] if m["name"] in sset_names)
                        if free_res >= req:
                            for sname in sset_names:
                                if sname in remaining:
                                    placement[sname] = dev
                            remaining -= sset_names
                            free_res -= req
                            self.node_busy_resources[dev] += req
                            if not remaining:
                                return placement
        
        return placement if not remaining else None
    
    def _order_devices_by_fitness(self, community, gateway, app, entities, G):
        """Order devices by fitness (exec_time + net_time)."""
        fitness = {}
        total_work = sum(m.get("RAM", 1) * 10000 for m in app["module"])  # approx MIPS
        
        for dev in community:
            if dev not in entities:
                continue
            ipt = entities[dev].get("IPT", 100)
            exec_time = total_work / max(ipt, 1)
            try:
                net_time = nx.shortest_path_length(G, source=gateway, target=dev, weight='weight')
            except:
                net_time = 1e9
            fitness[dev] = exec_time + net_time
        
        return [d for d, _ in sorted(fitness.items(), key=operator.itemgetter(1))]
    
    def _build_app_graph(self, app):
        """Build directed graph from app definition."""
        G = nx.DiGraph()
        for module in app["module"]:
            G.add_node(module["id"])
        
        for msg in app["message"]:
            if msg["s"] != "None":
                # Find source and dest module IDs
                src_id = next((m["id"] for m in app["module"] if m["name"] == msg["s"]), None)
                dst_id = next((m["id"] for m in app["module"] if m["name"] == msg["d"]), None)
                if src_id is not None and dst_id is not None:
                    G.add_edge(src_id, dst_id)
        
        return G
    
    def _get_app_gateways(self, users):
        """Extract gateways per app from users."""
        app_gw = {}
        for src in users.get("sources", []):
            app_id = src["app"]
            gw = src["id_resource"]
            if app_id not in app_gw:
                app_gw[app_id] = []
            if gw not in app_gw[app_id]:
                app_gw[app_id].append(gw)
        return app_gw
