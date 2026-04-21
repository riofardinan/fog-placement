"""
Particle Swarm Optimization based Placement (PSOPlacement)

Placement-style adaptation of `PSOoptimization` from `exp/algo 1 fitness`
using the JSON-based topology/app/users interface from this repository.

Key properties:
- Chromosome / particle: [device_id per service]
- Objective: **single fitness = normalized average network latency**
  across (app, gateway, service) triples, with hard RAM constraints.
"""

import random
from typing import Dict, List, Tuple

import networkx as nx

from placements.placement import Placement


class PSOPlacement(Placement):
    """
    PSO-based placement focusing on network latency (single objective).
    """

    def __init__(
        self,
        num_particles: int = 30,
        max_iter: int = 60,
        w: float = 0.7,
        c1: float = 1.5,
        c2: float = 1.5,
    ) -> None:
        super().__init__()
        self.name = "PSOPlacement"
        self.num_particles = num_particles
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

        # Internal structures (set in generate_allocation)
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

    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation with PSO metaheuristic.
        """
        # Build environment
        G = self._build_graph(topology)
        entities = {e["id"]: e for e in topology["entity"]}
        self._build_services(applications)
        self._build_devices(G, entities)
        self._build_app_mappings(applications, users)
        self._build_distance_cache(G)

        # Initialize swarm
        particles = [
            [random.choice(self.fog_nodes) for _ in range(self.num_services)]
            for _ in range(self.num_particles)
        ]
        velocities = [
            [0.0 for _ in range(self.num_services)]
            for _ in range(self.num_particles)
        ]

        pbest = [p[:] for p in particles]
        pbest_score = [self._evaluate(p) for p in particles]

        gbest = pbest[pbest_score.index(min(pbest_score))][:]
        gbest_score = min(pbest_score)

        # PSO iterations
        for _ in range(self.max_iter):
            for i in range(self.num_particles):
                score = self._evaluate(particles[i])

                if score < pbest_score[i]:
                    pbest[i] = particles[i][:]
                    pbest_score[i] = score

                if score < gbest_score:
                    gbest = particles[i][:]
                    gbest_score = score

            # Update velocity & position
            for i in range(self.num_particles):
                for s in range(self.num_services):
                    r1 = random.random()
                    r2 = random.random()

                    current_index = self.fog_nodes.index(particles[i][s])
                    pbest_index = self.fog_nodes.index(pbest[i][s])
                    gbest_index = self.fog_nodes.index(gbest[s])

                    velocities[i][s] = (
                        self.w * velocities[i][s]
                        + self.c1 * r1 * (pbest_index - current_index)
                        + self.c2 * r2 * (gbest_index - current_index)
                    )

                    new_index = int(round(current_index + velocities[i][s]))
                    new_index = max(0, min(len(self.fog_nodes) - 1, new_index))
                    particles[i][s] = self.fog_nodes[new_index]

        # Build allocation from gbest
        allocation: List[dict] = []
        for sid in range(self.num_services):
            allocation.append(
                {
                    "module_name": self.services[sid],
                    "app": str(self.service_to_app[sid]),
                    "id_resource": gbest[sid],
                }
            )

        # Mandatory cloud replica per service: use first cloud-like node if present,
        # otherwise reuse the assigned device.
        cloud_id = self._find_cloud_id(entities)
        for sid in range(self.num_services):
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

        # Representative packet size, aligned with other placements
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
        # Gateways per app
        self.app_gateways = {}
        for src in users.get("sources", []):
            app_id = src["app"]
            gw = src["id_resource"]
            self.app_gateways.setdefault(app_id, [])
            if gw not in self.app_gateways[app_id]:
                self.app_gateways[app_id].append(gw)

        # Ensure every app has at least one gateway (fallback)
        for app in applications:
            app_id = app["id"]
            if app_id not in self.app_gateways:
                self.app_gateways[app_id] = [self.fog_nodes[0]] if self.fog_nodes else []

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
                dist = lengths.get(fog, 1e12)
                self.distance_cache[(gw, fog)] = dist
                if dist < 1e11 and dist > self.max_latency:
                    self.max_latency = dist

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _evaluate(self, placement: List[int]) -> float:
        """
        Cost function (to be minimized):
        - Hard RAM constraint: infeasible => large cost
        - Average normalized latency over (app, gateway, service) tuples.
        """
        # RAM usage check
        usage: Dict[int, float] = {d: 0.0 for d in self.fog_nodes}

        for sid, dev in enumerate(placement):
            ram = self.service_ram.get(sid, 0.0)
            usage[dev] = usage.get(dev, 0.0) + ram
            if usage[dev] > self.node_ram_cap.get(dev, 0.0):
                # Heavy penalty for infeasible solution
                return 1e12

        # Latency term
        total = 0.0
        count = 0
        for app_id, sids in self.app_service_indices.items():
            gws = self.app_gateways.get(app_id, [])
            if not gws:
                continue
            for gw in gws:
                for sid in sids:
                    dev = placement[sid]
                    total += self.distance_cache.get((gw, dev), self.max_latency)
                    count += 1

        if count == 0:
            avg_latency = self.max_latency
        else:
            avg_latency = total / count

        norm_latency = avg_latency / max(self.max_latency, 1.0)
        return norm_latency

