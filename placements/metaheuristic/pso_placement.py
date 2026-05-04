"""
Particle Swarm Optimization based Placement (PSOPlacement)

Placement-style adaptation of `PSOoptimization` from `exp/algo 1 fitness`
using the JSON-based topology/app/users interface from this repository.

Key properties:
- Chromosome / particle: [fog_node_id per service]
- Objective: **single objective proxy of response time**
  (compute + hop * PR), with RAM validity bonus/penalty.

Notes:
- Cloud node is NOT a candidate placement target for PSO (consistent with GA/SM/FFHA).
- Do not emit duplicate allocations for the same (app, module): JSONPlacement/YAFS may
  treat later entries as overriding, which previously collapsed solutions to cloud.
"""

import random
from typing import Dict, List, Tuple

import networkx as nx

from placements.placement import Placement
from placements.metaheuristic._common import build_problem, fitness_response_proxy, greedy_seed_chrom, random_chrom, to_allocation


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
        self.node_to_index: Dict[int, int] = {}
        self.num_services: int = 0
        self.max_penalty: float = 1e18

    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation with PSO metaheuristic.
        """
        prob = build_problem(topology, applications, users)
        # candidate nodes include cloud as overflow (helps compete with SM/FFHA)
        self.fog_nodes = list(prob.candidate_nodes)
        self.node_to_index = {n: i for i, n in enumerate(self.fog_nodes)}
        self.num_services = len(prob.services)

        if self.num_services == 0 or not self.fog_nodes:
            return []

        # --- Initialize swarm (mix greedy seed + random) ---
        particles: List[List[int]] = []
        particles.append(greedy_seed_chrom(prob))
        while len(particles) < self.num_particles:
            particles.append(random_chrom(prob))

        velocities: List[List[float]] = [
            [0.0 for _ in range(self.num_services)] for _ in range(self.num_particles)
        ]

        # personal/global bests (minimize cost)
        pbest = [p[:] for p in particles]
        pbest_cost = [self._cost(p, prob) for p in particles]

        best_i = min(range(len(pbest_cost)), key=lambda i: pbest_cost[i])
        gbest = pbest[best_i][:]
        gbest_cost = pbest_cost[best_i]

        # --- PSO iterations ---
        for _ in range(self.max_iter):
            # evaluate and update bests
            for i in range(self.num_particles):
                cost = self._cost(particles[i], prob)
                if cost < pbest_cost[i]:
                    pbest[i] = particles[i][:]
                    pbest_cost[i] = cost
                if cost < gbest_cost:
                    gbest = particles[i][:]
                    gbest_cost = cost

            # update velocity & position (discrete via index space)
            for i in range(self.num_particles):
                for s in range(self.num_services):
                    r1 = random.random()
                    r2 = random.random()

                    current_idx = self.node_to_index[particles[i][s]]
                    pbest_idx = self.node_to_index[pbest[i][s]]
                    gbest_idx = self.node_to_index[gbest[s]]

                    velocities[i][s] = (
                        self.w * velocities[i][s]
                        + self.c1 * r1 * (pbest_idx - current_idx)
                        + self.c2 * r2 * (gbest_idx - current_idx)
                    )

                    new_idx = int(round(current_idx + velocities[i][s]))
                    new_idx = max(0, min(len(self.fog_nodes) - 1, new_idx))
                    particles[i][s] = self.fog_nodes[new_idx]

        return to_allocation(gbest, prob)

    # ------------------------------------------------------------------
    # Evaluation (minimize cost)
    # ------------------------------------------------------------------
    def _cost(self, chrom: List[int], prob) -> float:
        """
        Minimize cost = proxy_response_time (ms-ish).

        We reuse the shared proxy fitness (higher is better) by converting:
          cost = -fitness_response_proxy
        """
        try:
            fit = float(fitness_response_proxy(chrom, prob))
        except Exception:
            return self.max_penalty
        # fit is "maximize": larger is better => minimize negative.
        cost = -fit
        if cost != cost:  # NaN guard
            return self.max_penalty
        return cost

