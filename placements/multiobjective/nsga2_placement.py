"""
NSGA-II Placement (multi-objective).

Objectives (minimize):
1) total hops across all application chains
2) invalidity (0 valid, 1 invalid)

Produces a single allocation by selecting the best valid, lowest-hop solution
from the final population.
"""

from __future__ import annotations

import random

from placements.placement import Placement
from placements.metaheuristic._common import build_problem, random_chrom, to_allocation
from ._mo_common import (
    crowding_distance,
    fast_non_dominated_sort,
    crossover_one_point,
    mutate,
    pick_best_by_scalar,
)


class NSGAIIPlacement(Placement):
    def __init__(
        self,
        pop_size: int = 60,
        generations: int = 120,
        cx_rate: float = 0.7,
        mut_rate: float = 0.2,
        seed: int | None = None,
    ):
        super().__init__()
        self.name = "NSGAIIPlacement"
        self.pop_size = int(pop_size)
        self.generations = int(generations)
        self.cx_rate = float(cx_rate)
        self.mut_rate = float(mut_rate)
        self.seed = seed

    def generate_allocation(self, topology, applications, users):
        if self.seed is not None:
            random.seed(self.seed)

        prob = build_problem(topology, applications, users)
        if not prob.services or not prob.candidate_nodes:
            return []

        pop = [random_chrom(prob) for _ in range(self.pop_size)]

        for _ in range(self.generations):
            # Create offspring
            offspring = []
            while len(offspring) < self.pop_size:
                p1 = random.choice(pop)
                p2 = random.choice(pop)
                if random.random() < self.cx_rate:
                    c1, c2 = crossover_one_point(p1, p2)
                else:
                    c1, c2 = list(p1), list(p2)
                offspring.append(mutate(c1, prob, p=self.mut_rate))
                if len(offspring) < self.pop_size:
                    offspring.append(mutate(c2, prob, p=self.mut_rate))

            combined = pop + offspring
            fronts, objs = fast_non_dominated_sort(combined, prob)

            new_pop = []
            for front in fronts:
                if len(new_pop) + len(front) <= self.pop_size:
                    new_pop.extend([combined[i] for i in front])
                else:
                    cd = crowding_distance(front, objs)
                    front_sorted = sorted(front, key=lambda i: cd.get(i, 0.0), reverse=True)
                    needed = self.pop_size - len(new_pop)
                    new_pop.extend([combined[i] for i in front_sorted[:needed]])
                    break

            pop = new_pop

        best = pick_best_by_scalar(pop, prob)
        return to_allocation(best, prob)

