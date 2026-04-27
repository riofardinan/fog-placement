"""
Grey Wolf Optimizer (GWO) Placement.

This is a discrete/lightweight variant:
- population of chromosomes
- update step: wolves move toward alpha/beta/delta by copying genes with probabilities
- mutation: small random changes to keep exploration
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


class GWOPlacement(Placement):
    def __init__(
        self,
        wolves: int = 30,
        iterations: int = 150,
        copy_prob: float = 0.6,
        mutation_prob: float = 0.15,
        seed: int | None = None,
    ):
        super().__init__()
        self.name = "GWOPlacement"
        self.wolves = int(wolves)
        self.iterations = int(iterations)
        self.copy_prob = float(copy_prob)
        self.mutation_prob = float(mutation_prob)
        self.seed = seed

    def generate_allocation(self, topology, applications, users):
        if self.seed is not None:
            random.seed(self.seed)

        prob = build_problem(topology, applications, users)
        if not prob.services or not prob.fog_nodes:
            return []

        pop = [greedy_seed_chrom(prob)] + [random_chrom(prob) for _ in range(self.wolves - 1)]

        def rank():
            scored = [(fitness_response_proxy(c, prob), c) for c in pop]
            scored.sort(key=lambda x: x[0], reverse=True)
            return scored

        scored = rank()
        alpha = list(scored[0][1])
        beta = list(scored[1][1]) if len(scored) > 1 else list(alpha)
        delta = list(scored[2][1]) if len(scored) > 2 else list(beta)

        best = list(alpha)
        fbest = scored[0][0]

        for _ in range(self.iterations):
            scored = rank()
            alpha = list(scored[0][1])
            beta = list(scored[1][1]) if len(scored) > 1 else list(alpha)
            delta = list(scored[2][1]) if len(scored) > 2 else list(beta)

            if scored[0][0] > fbest:
                best = list(alpha)
                fbest = scored[0][0]

            new_pop = []
            for c in pop:
                child = list(c)
                for i in range(len(child)):
                    if random.random() < self.copy_prob:
                        # follow one of the leaders
                        leader = alpha if random.random() < 0.5 else (beta if random.random() < 0.7 else delta)
                        child[i] = leader[i]
                if random.random() < self.mutation_prob:
                    child = mutate_one_gene(child, prob)
                new_pop.append(child)
            pop = new_pop

        return to_allocation(best, prob)

