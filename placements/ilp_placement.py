"""
Integer Linear Programming based Placement (ILPPlacement)
Based on Lera et al., IEEE IoT Journal 2019
Minimizes network latency with capacity constraints.
"""
import itertools
import operator
import networkx as nx
from placements.placement import Placement
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False


class ILPPlacement(Placement):
    """
    ILP-based placement:
    - Objective: minimize Σ x[us,d] * latency(gateway, device)
    - Variables: x[(gateway, service), device] binary
    - Constraints: assignment + device capacity
    - Solved per app (sorted by deadline)
    """
    
    def __init__(self):
        super().__init__()
        self.name = "ILPPlacement"
        self.network_distances = {}
    
    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation using ILP optimization.
        
        Returns:
            List of allocation dicts
        """
        if not PULP_AVAILABLE:
            print("[ILP] Warning: PuLP not available, using greedy fallback")
            return self._greedy_fallback(topology, applications, users)
        
        # Build graph
        G = self._build_graph(topology)
        entities = {e["id"]: e for e in topology["entity"]}
        
        # Find cloud (YAFS: type CLOUD; or model cloud for backward compat)
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
        
        # Prepare user-service pairs
        user_services = []
        all_gateways = set()
        app_gw = {}
        
        for src in users.get("sources", []):
            app_id = src["app"]
            gw = src["id_resource"]
            all_gateways.add(gw)
            if app_id not in app_gw:
                app_gw[app_id] = []
            if gw not in app_gw[app_id]:
                app_gw[app_id].append(gw)
        
        for app in applications:
            gws = app_gw.get(app["id"], [1])
            for gw in gws:
                for module in app["module"]:
                    user_services.append((gw, module["name"], app["id"]))
        
        # Compute network distances
        self._compute_network_distances(G, all_gateways, cloud_id)
        
        # Device list (fog + cloud)
        fog_nodes = list(G.nodes())
        
        # Service resources
        service_ram = {}
        service_to_app = {}
        for app in applications:
            for module in app["module"]:
                service_ram[module["name"]] = module.get("RAM", 1)
                service_to_app[module["name"]] = app["id"]
        
        # Device capacities (mutable)
        device_caps = {n: entities[n].get("RAM", 1e9) for n in fog_nodes}
        device_caps[cloud_id] = 1e18  # unlimited cloud
        
        allocation = []
        
        # Sort apps by deadline
        sorted_apps = sorted(applications, key=lambda a: a.get("MaxLatency", 10000))
        
        for app in sorted_apps:
            app_id = app["id"]
            
            # Filter user-services for this app
            us_app = [(gw, sname) for (gw, sname, aid) in user_services if aid == app_id]
            
            if not us_app:
                continue
            
            # Assignment combinations
            assign_comb = list(itertools.product(us_app, fog_nodes + [cloud_id]))
            
            # Create ILP problem
            problem = pulp.LpProblem(f'ILP_APP_{app_id}', pulp.LpMinimize)
            
            x = {c: pulp.LpVariable(f'x_{c[0][0]}_{c[0][1]}_{c[1]}', cat='Binary')
                 for c in assign_comb}
            
            # Objective: minimize network latency
            problem += pulp.lpSum(
                x[c] * self._network_delay(c[0][0], c[1]) for c in assign_comb
            )
            
            # Constraint: each user-service assigned exactly once
            for us in us_app:
                problem += pulp.lpSum(
                    x[(us, d)] for d in fog_nodes + [cloud_id]
                ) == 1
            
            # Constraint: device capacity
            for d in fog_nodes + [cloud_id]:
                problem += pulp.lpSum(
                    x[(us, d)] * service_ram[us[1]] for us in us_app
                ) <= device_caps[d]
            
            # Solve
            problem.solve(pulp.PULP_CBC_CMD(msg=False))
            
            # Read solution
            if pulp.LpStatus[problem.status] == 'Optimal':
                for (us, d), var in x.items():
                    if var.value() == 1:
                        sname = us[1]
                        allocation.append({
                            "module_name": sname,
                            "app": str(app_id),
                            "id_resource": d
                        })
                        device_caps[d] -= service_ram[sname]
        
        return allocation
    
    def _build_graph(self, topology):
        """Build NetworkX graph."""
        G = nx.Graph()
        for entity in topology["entity"]:
            G.add_node(entity["id"], **entity)
        for link in topology["link"]:
            weight = link["PR"] + 2500000 / link["BW"]
            G.add_edge(link["s"], link["d"], weight=weight, **link)
        return G
    
    def _compute_network_distances(self, G, gateways, cloud_id):
        """Compute shortest path distances."""
        for gw in gateways:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, gw, weight='weight')
            except:
                lengths = {}
            for node in G.nodes():
                self.network_distances[(gw, node)] = lengths.get(node, 1e9)
            # Cloud penalty
            self.network_distances[(gw, cloud_id)] = 1e18
    
    def _network_delay(self, gateway, device):
        """Get network delay."""
        return self.network_distances.get((gateway, device), 1e9)
    
    def _greedy_fallback(self, topology, applications, users):
        """Greedy fallback when PuLP not available."""
        entities = {e["id"]: e for e in topology["entity"]}
        nodes = list(entities.keys())
        
        # Find cloud (YAFS: type CLOUD; or model cloud for backward compat)
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
        
        node_usage = {n: 0 for n in nodes}
        allocation = []
        
        # Sort apps by deadline
        sorted_apps = sorted(applications, key=lambda a: a.get("MaxLatency", 10000))
        
        for app in sorted_apps:
            modules = sorted(app["module"], key=lambda m: m.get("RAM", 0), reverse=True)
            for module in modules:
                best_node = None
                best_avail = -1
                
                for node_id in nodes:
                    cap = entities[node_id].get("RAM", 1e9)
                    avail = cap - node_usage[node_id]
                    if avail >= module.get("RAM", 0) and avail > best_avail:
                        best_node = node_id
                        best_avail = avail
                
                if best_node is None:
                    best_node = cloud_id
                
                allocation.append({
                    "module_name": module["name"],
                    "app": str(app["id"]),
                    "id_resource": best_node
                })
                node_usage[best_node] += module.get("RAM", 0)
        
        return allocation
