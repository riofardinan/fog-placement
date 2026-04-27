"""
Genetic Algorithm Placement (GAPlacement) — Pakpahan et al. (2025) Algorithm 6.

Chromosome : list[int]  — fog node ID per module (index = module index)
Fitness    : maximize  1/(total_hops + ε) + (0.3 if valid else −0.3)
             total_hops = sum of hop counts between consecutive chain nodes
             valid      = no fog node exceeds its RAM capacity
Selection  : tournament (k=3)
Crossover  : single-point, probability 0.5
Mutation   : per-gene uniform, probability 0.3 → reassign to random fog node
Elitism    : best individual survives every generation
Parameters : pop=100, generations=250
"""
import random
import networkx as nx
from placements.placement import Placement


class GAPlacement(Placement):

    def __init__(self, pop_size=100, generations=250, tourn_k=3,
                 cx_rate=0.5, mut_rate=0.3, seed=None):
        super().__init__()
        self.name = "GAPlacement"
        self.pop_size = pop_size
        self.generations = generations
        self.tourn_k = tourn_k
        self.cx_rate = cx_rate
        self.mut_rate = mut_rate
        self.seed = seed

    def generate_allocation(self, topology, applications, users):
        if self.seed is not None:
            random.seed(self.seed)

        entities = {e["id"]: e for e in topology["entity"]}
        cloud_id = self._find_cloud_id(entities)

        # Unweighted graph for hop counts
        G = nx.Graph()
        for e in topology["entity"]:
            G.add_node(e["id"])
        for link in topology["link"]:
            G.add_edge(link["s"], link["d"])

        fog_nodes = [n for n in G.nodes() if n != cloud_id]
        node_ram = {n: float(entities[n].get("RAM", 0)) for n in fog_nodes}

        # Flat service list across all apps
        services, service_to_app, service_ram = [], {}, {}
        for app in applications:
            for mod in app.get("module", []):
                idx = len(services)
                services.append(mod["name"])
                service_to_app[idx] = app["id"]
                service_ram[idx] = float(mod.get("RAM", 1))

        if not services or not fog_nodes:
            return []

        name_to_idx = {name: i for i, name in enumerate(services)}

        # Chain info: (list of service indices in chain order, source gateway node)
        chains_info = []
        for app in applications:
            chain = self._get_module_chain(app)
            source = self._get_app_source_node(app["id"], users) or fog_nodes[0]
            indices = [name_to_idx[n] for n in chain if n in name_to_idx]
            chains_info.append((indices, source))

        # Precompute all-pairs hop distances (unweighted)
        hop_dist = dict(nx.all_pairs_shortest_path_length(G))

        n = len(services)
        population = [[random.choice(fog_nodes) for _ in range(n)]
                      for _ in range(self.pop_size)]
        fitnesses = [self._fitness(c, chains_info, hop_dist, node_ram, service_ram, cloud_id)
                     for c in population]

        best_idx = max(range(self.pop_size), key=lambda i: fitnesses[i])
        best = list(population[best_idx])
        best_fit = fitnesses[best_idx]

        for _ in range(self.generations):
            new_pop = [list(best)]  # elitism
            while len(new_pop) < self.pop_size:
                p1 = self._tournament(population, fitnesses)
                p2 = self._tournament(population, fitnesses)
                c1, c2 = self._crossover(p1, p2)
                self._mutate(c1, fog_nodes)
                self._mutate(c2, fog_nodes)
                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = new_pop
            fitnesses = [self._fitness(c, chains_info, hop_dist, node_ram, service_ram, cloud_id)
                         for c in population]
            idx = max(range(self.pop_size), key=lambda i: fitnesses[i])
            if fitnesses[idx] > best_fit:
                best_fit = fitnesses[idx]
                best = list(population[idx])

        return [
            {"module_name": services[i], "app": str(service_to_app[i]), "id_resource": best[i]}
            for i in range(n)
        ]

    # ------------------------------------------------------------------
    # Fitness
    # ------------------------------------------------------------------

    def _fitness(self, chrom, chains_info, hop_dist, node_ram, service_ram, cloud_id):
        total_hops = 0
        for indices, source in chains_info:
            prev = source
            for idx in indices:
                curr = chrom[idx]
                total_hops += hop_dist.get(prev, {}).get(curr, 100)
                prev = curr

        node_load = {}
        for i, node in enumerate(chrom):
            if node != cloud_id:
                node_load[node] = node_load.get(node, 0.0) + service_ram[i]
        valid = all(load <= node_ram.get(n, 0) for n, load in node_load.items())

        return 1.0 / (total_hops + 1e-6) + (0.3 if valid else -0.3)

    # ------------------------------------------------------------------
    # GA operators
    # ------------------------------------------------------------------

    def _tournament(self, population, fitnesses):
        candidates = random.sample(range(len(population)), min(self.tourn_k, len(population)))
        return list(population[max(candidates, key=lambda i: fitnesses[i])])

    def _crossover(self, p1, p2):
        if random.random() > self.cx_rate or len(p1) < 2:
            return list(p1), list(p2)
        cut = random.randint(1, len(p1) - 1)
        return p1[:cut] + p2[cut:], p2[:cut] + p1[cut:]

    def _mutate(self, chrom, fog_nodes):
        for i in range(len(chrom)):
            if random.random() < self.mut_rate:
                possible_nodes = [n for n in fog_nodes if n != chrom[i]]
                if possible_nodes:
                    chrom[i] = random.choice(possible_nodes)
