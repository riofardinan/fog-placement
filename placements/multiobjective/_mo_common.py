from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from placements.metaheuristic._common import (
    PlacementProblem,
    build_problem,
    evaluate_hops,
    evaluate_ram_valid,
    mutate_one_gene,
    random_chrom,
)


@dataclass(frozen=True)
class Objectives:
    hops: int
    invalid: int  # 0 if valid, 1 if invalid


def objectives(chrom: List[int], prob: PlacementProblem) -> Objectives:
    return Objectives(
        hops=evaluate_hops(chrom, prob),
        invalid=0 if evaluate_ram_valid(chrom, prob) else 1,
    )


def dominates(a: Objectives, b: Objectives) -> bool:
    """Minimize both objectives."""
    return (a.hops <= b.hops and a.invalid <= b.invalid) and (a.hops < b.hops or a.invalid < b.invalid)


def fast_non_dominated_sort(pop: List[List[int]], prob: PlacementProblem):
    """Returns list of fronts, each front is list of indices."""
    objs = [objectives(c, prob) for c in pop]
    S = [set() for _ in pop]
    n = [0 for _ in pop]
    fronts: List[List[int]] = [[]]

    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                S[p].add(q)
            elif dominates(objs[q], objs[p]):
                n[p] += 1
        if n[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1], objs


def crowding_distance(front: List[int], objs: List[Objectives]):
    """Simple crowding distance for two objectives."""
    if not front:
        return {}
    dist = {i: 0.0 for i in front}

    # objective 1: hops
    front_sorted = sorted(front, key=lambda i: objs[i].hops)
    dist[front_sorted[0]] = dist[front_sorted[-1]] = float("inf")
    minv, maxv = objs[front_sorted[0]].hops, objs[front_sorted[-1]].hops
    denom = (maxv - minv) or 1.0
    for k in range(1, len(front_sorted) - 1):
        prevv = objs[front_sorted[k - 1]].hops
        nextv = objs[front_sorted[k + 1]].hops
        dist[front_sorted[k]] += (nextv - prevv) / denom

    # objective 2: invalid
    front_sorted = sorted(front, key=lambda i: objs[i].invalid)
    dist[front_sorted[0]] = dist[front_sorted[-1]] = float("inf")
    minv, maxv = objs[front_sorted[0]].invalid, objs[front_sorted[-1]].invalid
    denom = (maxv - minv) or 1.0
    for k in range(1, len(front_sorted) - 1):
        prevv = objs[front_sorted[k - 1]].invalid
        nextv = objs[front_sorted[k + 1]].invalid
        dist[front_sorted[k]] += (nextv - prevv) / denom

    return dist


def pick_best_by_scalar(pop: List[List[int]], prob: PlacementProblem) -> List[int]:
    """
    Convert Pareto set to single choice for simulator:
    - prioritize valid (invalid=0)
    - then minimize hops
    """
    best = None
    best_obj = None
    for c in pop:
        o = objectives(c, prob)
        if best is None:
            best, best_obj = c, o
            continue
        if o.invalid < best_obj.invalid or (o.invalid == best_obj.invalid and o.hops < best_obj.hops):
            best, best_obj = c, o
    return list(best)


def crossover_one_point(a: List[int], b: List[int]) -> Tuple[List[int], List[int]]:
    if len(a) < 2:
        return list(a), list(b)
    cut = random.randint(1, len(a) - 1)
    return a[:cut] + b[cut:], b[:cut] + a[cut:]


def mutate(chrom: List[int], prob: PlacementProblem, p: float = 0.2) -> List[int]:
    if random.random() < p:
        return mutate_one_gene(list(chrom), prob)
    return list(chrom)

