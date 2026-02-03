"""
Genetic Algorithm based Placement (GAPlacement)
Minimizes weighted fitness: latency + utilization_penalty + risk
"""
import random
import networkx as nx
from collections import defaultdict
from placements.placement import Placement

class GAPlacement(Placement):
    """
    GA-based placement:
    - Chromosome: [device_id per service]
    - Fitness: α*latency + β*util_penalty + γ*risk (minimization)
    - Tournament selection, one-point crossover, mutation
    - Capacity repair
    """
    
    def __init__(self, pop_size=40, generations=60, tourn_k=3,
                 cx_rate=0.9, mut_rate=0.2, candidate_k=10, seed=None):
        super().__init__()
        self.name = "GAPlacement"
        self.pop_size = pop_size
        self.generations = generations
        self.tourn_k = tourn_k
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.candidate_k = candidate_k
        self.seed = seed
        
        # Fitness weights
        self.weights = {"lat": 0.4, "util": 0.3, "risk": 0.3}
    
    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation using GA optimization.
        
        Returns:
            List of allocation dicts
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        # Build structures
        G = self._build_graph(topology)
        entities = {e["id"]: e for e in topology["entity"]}
        
        # Find cloud (YAFS: type CLOUD; or model cloud for backward compat)
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
                break
        
        # Service list
        services = []
        service_to_app = {}
        service_ram = {}
        for app in applications:
            for module in app["module"]:
                sid = len(services)
                services.append(module["name"])
                service_to_app[sid] = app["id"]
                service_ram[sid] = module.get("RAM", 1)
        
        nodes = list(G.nodes())
        
        # User gateways per app
        app_gw = self._get_app_gateways(users)
        
        # Precompute latencies
        latencies = self._precompute_latencies(G, app_gw, entities)
        
        # Build Top-K candidates per service
        candidates = self._build_candidates(services, service_to_app, app_gw, latencies, nodes, cloud_id)
        
        # GA Evolution
        population = [self._random_individual(services, candidates, cloud_id, nodes)
                      for _ in range(self.pop_size)]
        
        # Repair initial population
        for ind in population:
            self._repair_capacity(ind, services, service_ram, entities, cloud_id)
        
        # Evaluate
        fitnesses = [self._fitness(ind, services, service_to_app, service_ram, app_gw,
                                   latencies, entities, cloud_id)
                     for ind in population]
        
        best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
        best_ind = list(population[best_idx])
        best_fit = fitnesses[best_idx]
        
        # Evolve
        for gen in range(self.generations):
            new_pop = [list(best_ind)]  # Elitism
            
            while len(new_pop) < self.pop_size:
                p1 = self._tournament_select(population, fitnesses)
                p2 = self._tournament_select(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1, services, candidates, cloud_id, nodes)
                self._mutate(c2, services, candidates, cloud_id, nodes)
                self._repair_capacity(c1, services, service_ram, entities, cloud_id)
                self._repair_capacity(c2, services, service_ram, entities, cloud_id)
                new_pop.extend([c1, c2])
            
            population = new_pop[:self.pop_size]
            fitnesses = [self._fitness(ind, services, service_to_app, service_ram, app_gw,
                                       latencies, entities, cloud_id)
                         for ind in population]
            
            idx = min(range(len(population)), key=lambda i: fitnesses[i])
            if fitnesses[idx] < best_fit:
                best_fit = fitnesses[idx]
                best_ind = list(population[idx])
        
        # Build allocation from best individual
        allocation = []
        for sid, dev in enumerate(best_ind):
            allocation.append({
                "module_name": services[sid],
                "app": str(service_to_app[sid]),
                "id_resource": dev
            })
        
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
    
    def _get_app_gateways(self, users):
        """Get gateways per app."""
        app_gw = defaultdict(list)
        for src in users.get("sources", []):
            gw = src["id_resource"]
            if gw not in app_gw[src["app"]]:
                app_gw[src["app"]].append(gw)
        return dict(app_gw)
    
    def _precompute_latencies(self, G, app_gw, entities):
        """Precompute shortest path latencies."""
        latencies = {}
        all_gw = set()
        for gws in app_gw.values():
            all_gw.update(gws)
        
        for gw in all_gw:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, gw, weight='weight')
            except:
                lengths = {}
            for node in G.nodes():
                latencies[(gw, node)] = lengths.get(node, 1e9)
        
        return latencies
    
    def _build_candidates(self, services, service_to_app, app_gw, latencies, nodes, cloud_id):
        """Build Top-K candidates per service."""
        candidates = {}
        for sid in range(len(services)):
            app_id = service_to_app[sid]
            gws = app_gw.get(app_id, [1])
            
            avg_lat = []
            for node in nodes:
                lats = [latencies.get((gw, node), 1e9) for gw in gws]
                avg_lat.append((node, sum(lats) / len(lats)))
            
            avg_lat.sort(key=lambda x: x[1])
            topk = [n for n, _ in avg_lat[:self.candidate_k]]
            if cloud_id not in topk:
                topk.append(cloud_id)
            candidates[sid] = topk
        
        return candidates
    
    def _random_individual(self, services, candidates, cloud_id, nodes):
        """Create random individual."""
        ind = []
        for sid in range(len(services)):
            cand = candidates.get(sid, nodes)
            if cand:
                ind.append(random.choice(cand))
            elif cloud_id is not None:
                ind.append(cloud_id)
            else:
                ind.append(random.choice(nodes))
        return ind
    
    def _tournament_select(self, population, fitnesses):
        """Tournament selection."""
        k = max(2, self.tourn_k)
        best = None
        for _ in range(k):
            idx = random.randrange(len(population))
            if best is None or fitnesses[idx] < best[0]:
                best = (fitnesses[idx], population[idx])
        return list(best[1])
    
    def _crossover(self, p1, p2):
        """One-point crossover."""
        if random.random() > self.cx_rate or len(p1) < 2:
            return list(p1), list(p2)
        cut = random.randint(1, len(p1) - 1)
        return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]
    
    def _mutate(self, ind, services, candidates, cloud_id, nodes):
        """Mutate individual."""
        for sid in range(len(ind)):
            if random.random() < self.mut_rate:
                cand = candidates.get(sid, nodes)
                ind[sid] = random.choice(cand) if cand else cloud_id
    
    def _fitness(self, ind, services, service_to_app, service_ram, app_gw, latencies, entities, cloud_id):
        """Compute fitness (lower is better)."""
        eps = 1e-12
        
        # Utilization penalty
        demand = defaultdict(float)
        for sid, dev in enumerate(ind):
            demand[dev] += service_ram[sid]
        
        util_penalty = 0.0
        for dev, dem in demand.items():
            cap = entities.get(dev, {}).get("RAM", 1e9)
            if cap > 0:
                util_penalty += max(0.0, (dem - cap) / cap)
        
        # Latency
        lat_vals = []
        for sid, dev in enumerate(ind):
            app_id = service_to_app[sid]
            gws = app_gw.get(app_id, [1])
            for gw in gws:
                lat_vals.append(latencies.get((gw, dev), 1e9))
        
        avg_lat = (sum(lat_vals) / len(lat_vals)) if lat_vals else 0.0
        
        # Risk (simplified: assume availability = 1.0)
        avg_risk = 0.0
        
        # Normalize
        lat_norm = avg_lat / (1000.0 + eps)
        util_norm = util_penalty / (len(entities) + eps)
        risk_norm = avg_risk / (1.0 + eps)
        
        fitness = (
            self.weights["lat"] * lat_norm +
            self.weights["util"] * util_norm +
            self.weights["risk"] * risk_norm
        )
        return fitness
    
    def _repair_capacity(self, ind, services, service_ram, entities, cloud_id):
        """Repair capacity overload."""
        services_on = defaultdict(list)
        for sid, dev in enumerate(ind):
            services_on[dev].append(sid)
        
        for dev, sids in list(services_on.items()):
            cap = entities.get(dev, {}).get("RAM", 1e9)
            demand = sum(service_ram[s] for s in sids)
            
            while demand > cap + 1e-9 and dev != cloud_id:
                # Move largest service to cloud
                s_move = max(sids, key=lambda s: service_ram[s])
                sids.remove(s_move)
                ind[s_move] = cloud_id
                services_on[cloud_id].append(s_move)
                demand -= service_ram[s_move]
