"""
Complex Network + GA-PSO Hybrid Placement (CNGAPSOPlacement)

Placement-style adaptation of `CN_GAPSO` from `exp/algo 1 fitness`,
using JSON-based topology/app/users as in other placement classes.

Key properties (aligned with "algo 1 fitness"):
- Decision variable: [device_id per service]
- Objective: **single fitness = normalized network latency**
  over (app, gateway, service) tuples.
"""

import random
from typing import Dict, List, Tuple

import networkx as nx

from placements.placement import Placement


class CNGAPSOPlacement(Placement):
    """
    CN + GA-PSO hybrid placement focusing on latency.
    """

    def __init__(
        self,
        population_size: int = 50,
        generations: int = 100,
        mutation_rate: float = 0.2,
        tournament_size: int = 3,
        c1: float = 1.5,
        c2: float = 1.5,
        w: float = 0.7,
    ) -> None:
        super().__init__()
        self.name = "CNGAPSOPlacement"

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

        self.c1 = c1
        self.c2 = c2
        self.w = w

        # Environment (initialised in generate_allocation)
        self.fog_nodes: List[int] = []
        self.num_services: int = 0
        self.services: List[str] = []
        self.service_to_app: Dict[int, int] = {}
        self.service_ram: Dict[int, float] = {}
        self.node_ram_cap: Dict[int, float] = {}
        self.app_gateways: Dict[int, List[int]] = {}
        self.app_service_indices: Dict[int, List[int]] = {}
        self.distance_cache: Dict[Tuple[int, int], float] = {}
        self.max_latency: float = 1.0
        self.ranked_nodes: List[int] = []

    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation with CN-GA-PSO hybrid metaheuristic.
        """
        G = self._build_graph(topology)
        entities = {e["id"]: e for e in topology["entity"]}

        self._build_services(applications)
        self._build_devices(G, entities)
        self._build_app_mappings(applications, users)
        self._build_distance_cache(G)
        self._pre_process_nodes(G, entities)

        # Initial population (biased towards top-ranked nodes)
        population: List[List[int]] = []
        for _ in range(self.population_size):
            ind: List[int] = []
            for _ in range(self.num_services):
                if random.random() < 0.8 and self.ranked_nodes:
                    ind.append(random.choice(self.ranked_nodes))
                else:
                    ind.append(random.choice(self.fog_nodes))
            population.append(ind)

        fitness = [self._evaluate(ind) for ind in population]

        # Personal and global bests
        pbest = [p[:] for p in population]
        pbest_score = list(fitness)
        gbest = population[fitness.index(max(fitness))][:]
        gbest_score = max(fitness)

        for _ in range(self.generations):
            new_pop: List[List[int]] = []
            while len(new_pop) < self.population_size:
                p1 = self._selection(population, fitness)
                p2 = self._selection(population, fitness)

                pt = random.randint(1, self.num_services - 1)
                child = p1[:pt] + p2[pt:]

                # Mutation
                if random.random() < self.mutation_rate and self.ranked_nodes:
                    idx = random.randint(0, self.num_services - 1)
                    child[idx] = random.choice(self.ranked_nodes)

                # Hybrid PSO-style refinement towards p1 and gbest
                for s in range(self.num_services):
                    r_pso = random.random()
                    if r_pso < 0.1:
                        child[s] = gbest[s]
                    elif r_pso < 0.2:
                        child[s] = p1[s]

                new_pop.append(child)

            population = new_pop
            fitness = [self._evaluate(ind) for ind in population]

            for i in range(self.population_size):
                if fitness[i] > pbest_score[i]:
                    pbest[i] = population[i][:]
                    pbest_score[i] = fitness[i]

                if fitness[i] > gbest_score:
                    gbest = population[i][:]
                    gbest_score = fitness[i]

        # Build allocation from gbest
        allocation: List[dict] = []
        cloud_id = self._find_cloud_id(entities)

        for sid in range(self.num_services):
            allocation.append(
                {
                    "module_name": self.services[sid],
                    "app": str(self.service_to_app[sid]),
                    "id_resource": gbest[sid],
                }
            )
            allocation.append(
                {
                    "module_name": self.services[sid],
                    "app": str(self.service_to_app[sid]),
                    "id_resource": cloud_id,
                }
            )

        return allocation

    # ------------------------------------------------------------------
    # Environment builders
    # ------------------------------------------------------------------
    def _build_graph(self, topology):
        """Build NetworkX graph with latency weights (PR + size/BW)."""
        G = nx.Graph()
        for entity in topology["entity"]:
            G.add_node(entity["id"], **entity)

        default_size = 3_000_000.0
        for link in topology["link"]:
            weight = float(link["PR"]) + default_size / float(link["BW"])
            G.add_edge(link["s"], link["d"], weight=weight, **link)

        return G

    def _find_cloud_id(self, entities: Dict[int, dict]) -> int:
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
        return cloud_id

    def _build_services(self, applications):
        self.services = []
        self.service_to_app = {}
        self.service_ram = {}
        self.app_service_indices = {}

        for app in applications:
            app_id = app["id"]
            for module in app.get("module", []):
                sid = len(self.services)
                self.services.append(module["name"])
                self.service_to_app[sid] = app_id
                self.service_ram[sid] = float(module.get("RAM", 1))
                self.app_service_indices.setdefault(app_id, []).append(sid)

        self.num_services = len(self.services)

    def _build_devices(self, G, entities: Dict[int, dict]):
        self.fog_nodes = list(G.nodes())
        self.node_ram_cap = {
            n: float(entities.get(n, {}).get("RAM", 1e9)) for n in self.fog_nodes
        }

    def _build_app_mappings(self, applications, users):
        self.app_gateways = {}
        for src in users.get("sources", []):
            app_id = src["app"]
            gw = src["id_resource"]
            self.app_gateways.setdefault(app_id, [])
            if gw not in self.app_gateways[app_id]:
                self.app_gateways[app_id].append(gw)

        # Fallback gateway for apps without explicit sources
        for app in applications:
            app_id = app["id"]
            if app_id not in self.app_gateways and self.fog_nodes:
                self.app_gateways[app_id] = [self.fog_nodes[0]]

    def _build_distance_cache(self, G):
        self.distance_cache = {}
        all_gateways = set()
        for gws in self.app_gateways.values():
            all_gateways.update(gws)

        self.max_latency = 1.0
        for gw in all_gateways:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, gw, weight="weight")
            except Exception:
                lengths = {}
            for fog in self.fog_nodes:
                dist = lengths.get(fog, 1e11)
                self.distance_cache[(gw, fog)] = dist
                if dist < 1e10 and dist > self.max_latency:
                    self.max_latency = dist

    def _pre_process_nodes(self, G, entities: Dict[int, dict]):
        """
        Rank nodes using betweenness centrality and RAM capacity,
        similar to CN_GAPSO pre-processing.
        """
        centrality = nx.betweenness_centrality(G, weight="weight")
        scored_nodes: List[Tuple[int, float]] = []

        max_ram = max((entities.get(n, {}).get("RAM", 1.0) for n in self.fog_nodes), default=1.0)
        for node in self.fog_nodes:
            ram = entities.get(node, {}).get("RAM", 0.0)
            ram_score = ram / max_ram if max_ram > 0 else 0.0
            score = 0.7 * centrality.get(node, 0.0) + 0.3 * ram_score
            scored_nodes.append((node, score))

        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        top_k = int(len(scored_nodes) * 0.5) or len(scored_nodes)
        self.ranked_nodes = [n for n, _ in scored_nodes[:top_k]]
        if not self.ranked_nodes:
            self.ranked_nodes = list(self.fog_nodes)

    # ------------------------------------------------------------------
    # Evaluation & selection
    # ------------------------------------------------------------------
    def _evaluate(self, placement: List[int]) -> float:
        """
        Fitness (to be maximized):
        - Hard RAM constraint: infeasible => large negative value
        - - normalized latency (higher is better).
        """
        usage: Dict[int, float] = {d: 0.0 for d in self.fog_nodes}

        for sid, dev in enumerate(placement):
            ram = self.service_ram.get(sid, 0.0)
            usage[dev] = usage.get(dev, 0.0) + ram
            if usage[dev] > self.node_ram_cap.get(dev, 0.0):
                # Heavy penalty for infeasible solution
                return -1e12

        total_latency = 0.0
        req_count = 0
        for app_id, sids in self.app_service_indices.items():
            gws = self.app_gateways.get(app_id, [])
            if not gws:
                continue
            for gw in gws:
                for sid in sids:
                    dev = placement[sid]
                    total_latency += self.distance_cache.get((gw, dev), self.max_latency)
                    req_count += 1

        if req_count == 0:
            norm_lat = 1.0
        else:
            norm_lat = (total_latency / (req_count * max(self.max_latency, 1.0)))

        # Higher fitness is better, so we maximize negative latency
        return -norm_lat

    def _selection(self, population: List[List[int]], fitness: List[float]) -> List[int]:
        idx = random.sample(range(len(population)), max(2, self.tournament_size))
        best_idx = max(idx, key=lambda i: fitness[i])
        return population[best_idx][:]

