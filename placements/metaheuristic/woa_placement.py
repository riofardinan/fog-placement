"""
Whale Optimization Algorithm (WOA) Placement.

Discrete/lightweight variant (kept simple):
- population of chromosomes
- exploitation: copy segments from best
- exploration: random re-sampling of a few genes
- objective: GA-like scalar fitness
"""

from __future__ import annotations

import random

from placements.placement import Placement
from ._common import (
    build_problem,
    fitness_response_proxy,
    greedy_seed_chrom,
    mutate_one_gene,
    random_chrom,
    to_allocation,
)


class WOAPlacement(Placement):
    def __init__(
        self,
        whales: int = 30,
        iterations: int = 150,
        spiral_prob: float = 0.5,
        segment_copy_prob: float = 0.7,
        mutation_prob: float = 0.2,
        seed: int | None = None,
    ):
        super().__init__()
        self.name = "WOAPlacement"
        self.whales = int(whales)
        self.iterations = int(iterations)
        self.spiral_prob = float(spiral_prob)
        self.segment_copy_prob = float(segment_copy_prob)
        self.mutation_prob = float(mutation_prob)
        self.seed = seed

    def generate_allocation(self, topology, applications, users):
        if self.seed is not None:
            random.seed(self.seed)

        prob = build_problem(topology, applications, users)
        if not prob.services or not prob.fog_nodes:
            return []

        pop = [greedy_seed_chrom(prob)] + [random_chrom(prob) for _ in range(self.whales - 1)]
        best = None
        fbest = None

        for _ in range(self.iterations):
            scored = [(fitness_response_proxy(c, prob), c) for c in pop]
            scored.sort(key=lambda x: x[0], reverse=True)
            if fbest is None or scored[0][0] > fbest:
                fbest = scored[0][0]
                best = list(scored[0][1])

            new_pop = []
            for c in pop:
                child = list(c)
                if random.random() < self.spiral_prob:
                    # spiral: copy a random segment from best
                    if random.random() < self.segment_copy_prob and len(child) >= 2:
                        a = random.randint(0, len(child) - 2)
                        b = random.randint(a + 1, len(child) - 1)
                        child[a : b + 1] = best[a : b + 1]
                else:
                    # exploration: randomize a few genes
                    for _k in range(max(1, len(child) // 20)):
                        i = random.randrange(len(child))
                        child[i] = random.choice(prob.fog_nodes)

                if random.random() < self.mutation_prob:
                    child = mutate_one_gene(child, prob)
                new_pop.append(child)

            pop = new_pop

        return to_allocation(best, prob) if best is not None else []

