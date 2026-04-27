"""
Simulated Annealing (SA) Placement.

Lightweight implementation:
- state: chromosome (service -> fog node)
- objective: same scalar fitness as GA baseline (min hops + RAM validity bonus)
- neighbor: mutate one gene
"""

from __future__ import annotations

import math
import random

from placements.placement import Placement
from placements.metaheuristic._common import (
    build_problem,
    fitness_response_proxy,
    greedy_seed_chrom,
    mutate_one_gene,
    random_chrom,
    to_allocation,
)


class SAPlacement(Placement):
    def __init__(
        self,
        iterations: int = 4000,
        t0: float = 1.0,
        alpha: float = 0.999,
        seed: int | None = None,
    ):
        super().__init__()
        self.name = "SAPlacement"
        self.iterations = int(iterations)
        self.t0 = float(t0)
        self.alpha = float(alpha)
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

        T = max(self.t0, 1e-9)
        for _ in range(self.iterations):
            y = mutate_one_gene(list(x), prob)
            fy = fitness_response_proxy(y, prob)
            delta = fy - fx
            if delta >= 0:
                x, fx = y, fy
            else:
                # accept with probability exp(delta/T)
                if random.random() < math.exp(delta / max(T, 1e-12)):
                    x, fx = y, fy
            if fx > fbest:
                best, fbest = list(x), fx
            T *= self.alpha

        return to_allocation(best, prob)

