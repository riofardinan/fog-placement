"""
MOEA/D Placement (multi-objective, decomposition-based).

Objectives (minimize):
1) total hops
2) invalidity (0/1)

We use a small set of weight vectors and optimize Tchebycheff scalarization.
Kept intentionally lightweight.
"""

from __future__ import annotations

import random
from typing import List, Tuple

from placements.placement import Placement
from placements.metaheuristic._common import build_problem, mutate_one_gene, random_chrom, to_allocation
from ._mo_common import Objectives, objectives, pick_best_by_scalar


def tchebycheff(obj: Objectives, w: Tuple[float, float], z: Objectives) -> float:
    return max(w[0] * abs(obj.hops - z.hops), w[1] * abs(obj.invalid - z.invalid))


class MOEADPlacement(Placement):
    def __init__(
        self,
        subproblems: int = 20,
        iterations: int = 200,
        neighbor_size: int = 5,
        mutation_prob: float = 0.25,
        seed: int | None = None,
    ):
        super().__init__()
        self.name = "MOEADPlacement"
        self.subproblems = int(subproblems)
        self.iterations = int(iterations)
        self.neighbor_size = int(neighbor_size)
        self.mutation_prob = float(mutation_prob)
        self.seed = seed

    def generate_allocation(self, topology, applications, users):
        if self.seed is not None:
            random.seed(self.seed)

        prob = build_problem(topology, applications, users)
        if not prob.services or not prob.fog_nodes:
            return []

        # weight vectors along line w1+w2=1
        ws = [(i / (self.subproblems - 1), 1.0 - i / (self.subproblems - 1)) for i in range(self.subproblems)]
        pop = [random_chrom(prob) for _ in range(self.subproblems)]
        objs = [objectives(c, prob) for c in pop]

        # ideal point
        z = Objectives(hops=min(o.hops for o in objs), invalid=min(o.invalid for o in objs))

        # neighborhood by weight distance
        def wdist(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        neigh = []
        for i, wi in enumerate(ws):
            d = sorted([(wdist(wi, wj), j) for j, wj in enumerate(ws)])
            neigh.append([j for _dist, j in d[: self.neighbor_size]])

        for _ in range(self.iterations):
            for i in range(self.subproblems):
                # pick parents from neighborhood
                nbs = neigh[i]
                p1 = pop[random.choice(nbs)]
                p2 = pop[random.choice(nbs)]

                # recombination (uniform)
                child = list(p1)
                for g in range(len(child)):
                    if random.random() < 0.5:
                        child[g] = p2[g]
                if random.random() < self.mutation_prob:
                    child = mutate_one_gene(child, prob)

                o_child = objectives(child, prob)
                # update ideal point
                z = Objectives(hops=min(z.hops, o_child.hops), invalid=min(z.invalid, o_child.invalid))

                # update neighbors if improved
                for j in nbs:
                    if tchebycheff(o_child, ws[j], z) <= tchebycheff(objs[j], ws[j], z):
                        pop[j] = child
                        objs[j] = o_child

        best = pick_best_by_scalar(pop, prob)
        return to_allocation(best, prob)

