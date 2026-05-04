"""
Tabu Search (TS) Placement.

Lightweight implementation:
- state: chromosome (service -> fog node)
- move: mutate one gene (service i -> node j)
- tabu: keep last K moves (i,j) to prevent immediate cycling
- objective: same scalar fitness as GA baseline
"""

from __future__ import annotations

import random
from collections import deque

from placements.placement import Placement
from placements.metaheuristic._common import (
    build_problem,
    fitness_response_proxy,
    greedy_seed_chrom,
    random_chrom,
    to_allocation,
)


class TSPlacement(Placement):
    def __init__(
        self,
        iterations: int = 2500,
        neighbors_per_iter: int = 20,
        tabu_size: int = 200,
        seed: int | None = None,
    ):
        super().__init__()
        self.name = "TSPlacement"
        self.iterations = int(iterations)
        self.neighbors_per_iter = int(neighbors_per_iter)
        self.tabu_size = int(tabu_size)
        self.seed = seed

    def generate_allocation(self, topology, applications, users):
        if self.seed is not None:
            random.seed(self.seed)

        prob = build_problem(topology, applications, users)
        x = greedy_seed_chrom(prob) or random_chrom(prob)
        if not x:
            return []

        fx = fitness_response_proxy(x, prob)
        best, fbest = list(x), fx

        tabu = deque(maxlen=self.tabu_size)  # store (gene_idx, new_node)

        for _ in range(self.iterations):
            best_n = None
            fbest_n = None
            move_best = None

            for _k in range(self.neighbors_per_iter):
                i = random.randrange(len(x))
                options = [n for n in prob.candidate_nodes if n != x[i]]
                if not options:
                    continue
                new_node = random.choice(options)
                move = (i, new_node)

                y = list(x)
                y[i] = new_node
                fy = fitness_response_proxy(y, prob)

                # aspiration: allow tabu move if it improves global best
                if move in tabu and fy <= fbest:
                    continue

                if fbest_n is None or fy > fbest_n:
                    best_n = y
                    fbest_n = fy
                    move_best = move

            if best_n is None:
                continue

            x = best_n
            fx = float(fbest_n)
            if move_best is not None:
                tabu.append(move_best)

            if fx > fbest:
                best, fbest = list(x), fx

        return to_allocation(best, prob)

