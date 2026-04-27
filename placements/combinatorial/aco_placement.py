"""
Ant Colony Optimization (ACO) Placement.

Lightweight ACO:
- pheromone tau[i][node] for service i assigned to node
- heuristic eta is based on (1 / (hop_contrib + 1)) where hop_contrib uses current partial chain
- objective: maximize GA-like fitness (min hops + validity bonus)

Note: kept intentionally simple (no extra dependencies).
"""

from __future__ import annotations

import random
from typing import Dict, List

from placements.placement import Placement
from placements.metaheuristic._common import build_problem, fitness_response_proxy, greedy_seed_chrom, to_allocation


class ACOPlacement(Placement):
    def __init__(
        self,
        ants: int = 25,
        iterations: int = 120,
        alpha: float = 1.0,   # pheromone influence
        beta: float = 2.0,    # heuristic influence
        rho: float = 0.1,     # evaporation
        q: float = 1.0,       # deposit scale
        seed: int | None = None,
    ):
        super().__init__()
        self.name = "ACOPlacement"
        self.ants = int(ants)
        self.iterations = int(iterations)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.q = float(q)
        self.seed = seed

    def generate_allocation(self, topology, applications, users):
        if self.seed is not None:
            random.seed(self.seed)

        prob = build_problem(topology, applications, users)
        n = len(prob.services)
        if n == 0 or not prob.fog_nodes:
            return []

        # tau per gene per node
        tau: List[Dict[int, float]] = [{node: 1.0 for node in prob.fog_nodes} for _ in range(n)]

        best = None
        fbest = None

        for _it in range(self.iterations):
            solutions = []
            fitnesses = []

            for _a in range(self.ants):
                chrom = self._construct_solution(prob, tau)
                fit = fitness_response_proxy(chrom, prob)
                solutions.append(chrom)
                fitnesses.append(fit)

                if fbest is None or fit > fbest:
                    best = list(chrom)
                    fbest = float(fit)

            # evaporate
            for i in range(n):
                for node in prob.fog_nodes:
                    tau[i][node] *= (1.0 - self.rho)
                    if tau[i][node] < 1e-9:
                        tau[i][node] = 1e-9

            # deposit from best few
            ranked = sorted(range(len(solutions)), key=lambda k: fitnesses[k], reverse=True)
            topk = ranked[: max(1, len(ranked) // 5)]
            for k in topk:
                chrom = solutions[k]
                fit = fitnesses[k]
                deposit = self.q * max(fit, 0.0)
                for i, node in enumerate(chrom):
                    tau[i][node] += deposit

        return to_allocation(best, prob) if best is not None else []

    def _construct_solution(self, prob, tau):
        """
        Construct chromosome sequentially.
        Heuristic: prefer nodes that reduce hop contributions for chains encountered.
        """
        chrom = [None] * len(prob.services)

        # For heuristic, track last placed node per chain while building.
        # Build map from service index -> chain id and position
        svc_to_chain_pos = {}
        for ci, (indices, source) in enumerate(prob.chains_info):
            prev = source
            for pos, idx in enumerate(indices):
                svc_to_chain_pos[idx] = (ci, pos, prev)
                prev = idx  # note: not used further; we only need prev node id, set later

        last_node_for_chain = {ci: source for ci, (_indices, source) in enumerate(prob.chains_info)}

        # seed the first few genes using greedy IPT order to reduce random drift
        seed = greedy_seed_chrom(prob)
        for i in range(len(prob.services)):
            weights = []
            nodes = prob.fog_nodes

            # heuristic based on hop from last node in its chain (if applicable)
            ci = None
            if i in svc_to_chain_pos:
                ci = svc_to_chain_pos[i][0]

            for node in nodes:
                t = tau[i][node] ** self.alpha
                if ci is None:
                    eta = 1.0
                else:
                    prev_node = last_node_for_chain.get(ci, prob.fog_nodes[0])
                    hop = prob.hop_dist.get(prev_node, {}).get(node, 100)
                    # response proxy: hops*PR + inst/IPT (scaled into heuristic)
                    ipt = float(prob.node_ipt.get(node, 1.0)) or 1.0
                    inst = float(prob.service_inst.get(i, 0.0))
                    eta_cost = hop * prob.mean_pr_ms + (inst / ipt if inst > 0 else 0.0)
                    eta = (1.0 / (eta_cost + 1.0)) ** self.beta
                weights.append(t * eta)

            s = sum(weights)
            r = random.random() * s
            acc = 0.0
            chosen = nodes[-1]
            for node, w in zip(nodes, weights):
                acc += w
                if acc >= r:
                    chosen = node
                    break

            # blend with greedy seed early to stabilize
            if i < min(3, len(seed)) and random.random() < 0.5:
                chosen = seed[i]
            chrom[i] = chosen
            if ci is not None:
                last_node_for_chain[ci] = chosen

        return chrom

