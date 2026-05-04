from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx


@dataclass(frozen=True)
class PlacementProblem:
    services: List[str]                 # flat list across all apps
    service_to_app: Dict[int, str]      # idx -> app_id (string)
    service_ram: Dict[int, float]       # idx -> RAM demand
    chains_info: List[Tuple[List[int], int]]  # (service indices in chain order, source gateway node id)
    fog_nodes: List[int]                # candidate fog nodes (excludes cloud)
    candidate_nodes: List[int]          # fog + cloud (cloud acts as overflow)
    cloud_id: int
    node_ram: Dict[int, float]
    hop_dist: Dict[int, Dict[int, int]]  # hop distances
    node_ipt: Dict[int, float]
    service_inst: Dict[int, float]
    mean_pr_ms: float


def build_problem(topology, applications, users) -> PlacementProblem:
    """Build a lightweight optimization problem from scenario JSON."""
    entities = {e["id"]: e for e in topology["entity"]}
    cloud_id = _find_cloud_id(entities)

    # Unweighted hop graph
    G = nx.Graph()
    for e in topology["entity"]:
        G.add_node(e["id"])
    for link in topology["link"]:
        G.add_edge(link["s"], link["d"])

    fog_nodes = [n for n in G.nodes() if n != cloud_id]
    candidate_nodes = list(fog_nodes) + ([cloud_id] if cloud_id in G.nodes() else [])

    node_ram = {n: float(entities[n].get("RAM", 0)) for n in fog_nodes}
    # include cloud IPT too (cloud may be chosen as overflow)
    node_ipt = {n: float(entities.get(n, {}).get("IPT", 1.0)) for n in candidate_nodes}

    prs = [float(l.get("PR", 0.0)) for l in topology.get("link", []) if "PR" in l]
    mean_pr_ms = float(sum(prs) / len(prs)) if prs else 0.0

    services: List[str] = []
    service_to_app: Dict[int, str] = {}
    service_ram: Dict[int, float] = {}
    service_inst: Dict[int, float] = {}
    for app in applications:
        app_id = str(app["id"])
        for mod in app.get("module", []):
            idx = len(services)
            services.append(mod["name"])
            service_to_app[idx] = app_id
            service_ram[idx] = float(mod.get("RAM", 1))
            # Generator stores instruction size as "instructions" (uniform 20k–60k)
            service_inst[idx] = float(mod.get("instructions", mod.get("inst", 0.0)) or 0.0)

    name_to_idx = {name: i for i, name in enumerate(services)}

    chains_info: List[Tuple[List[int], int]] = []
    for app in applications:
        app_id = str(app["id"])
        chain = _get_module_chain(app)
        source = _get_app_source_node(app_id, users)
        if source is None:
            source = fog_nodes[0] if fog_nodes else cloud_id
        indices = [name_to_idx[n] for n in chain if n in name_to_idx]
        chains_info.append((indices, source))

    hop_dist = dict(nx.all_pairs_shortest_path_length(G))

    return PlacementProblem(
        services=services,
        service_to_app=service_to_app,
        service_ram=service_ram,
        chains_info=chains_info,
        fog_nodes=fog_nodes,
        candidate_nodes=candidate_nodes,
        cloud_id=cloud_id,
        node_ram=node_ram,
        hop_dist=hop_dist,
        node_ipt=node_ipt,
        service_inst=service_inst,
        mean_pr_ms=mean_pr_ms,
    )


def evaluate_hops(chrom: List[int], prob: PlacementProblem) -> int:
    """Total hops between source->first and consecutive modules across all apps."""
    total_hops = 0
    for indices, source in prob.chains_info:
        prev = source
        for idx in indices:
            curr = chrom[idx]
            total_hops += prob.hop_dist.get(prev, {}).get(curr, 100)
            prev = curr
    return int(total_hops)


def evaluate_ram_valid(chrom: List[int], prob: PlacementProblem) -> bool:
    node_load: Dict[int, float] = {}
    for i, node in enumerate(chrom):
        if node == prob.cloud_id:
            continue
        node_load[node] = node_load.get(node, 0.0) + float(prob.service_ram.get(i, 1.0))
    return all(load <= prob.node_ram.get(n, 0.0) for n, load in node_load.items())


def fitness_single_objective(chrom: List[int], prob: PlacementProblem) -> float:
    """
    Keep consistent with the paper's GA baseline:
      maximize 1/(total_hops+eps) + bonus(valid)/penalty(invalid)
    """
    hops = evaluate_hops(chrom, prob)
    valid = evaluate_ram_valid(chrom, prob)
    return 1.0 / (hops + 1e-6) + (0.3 if valid else -0.3)


def fitness_response_proxy(chrom: List[int], prob: PlacementProblem) -> float:
    """
    Lightweight proxy of paper Eq.(1–4) without queueing:
      response ≈ sum(inst/IPT) + sum(hops * PR)
    We maximize negative cost, plus validity bonus/penalty.
    """
    total_ms = 0.0
    for indices, source in prob.chains_info:
        prev = source
        for idx in indices:
            node = chrom[idx]
            hop = float(prob.hop_dist.get(prev, {}).get(node, 100))
            total_ms += hop * prob.mean_pr_ms
            ipt = float(prob.node_ipt.get(node, 1.0)) or 1.0
            inst = float(prob.service_inst.get(idx, 0.0))
            if inst > 0:
                total_ms += inst / ipt
            prev = node

    valid = evaluate_ram_valid(chrom, prob)
    # maximize: higher is better
    return -total_ms + (0.3 if valid else -0.3)


def greedy_seed_chrom(prob: PlacementProblem) -> List[int]:
    """Simple SM-like seed with cloud overflow."""
    if not prob.services or not prob.fog_nodes or prob.cloud_id is None:
        return []
    ranked = sorted(prob.fog_nodes, key=lambda n: prob.node_ipt.get(n, 0.0), reverse=True)
    caps = {n: float(prob.node_ram.get(n, 0.0)) for n in ranked}
    chrom: List[int] = []
    for i in range(len(prob.services)):
        ram = float(prob.service_ram.get(i, 1.0))
        chosen = prob.cloud_id
        for n in ranked:
            if caps.get(n, 0.0) >= ram:
                chosen = n
                caps[n] -= ram
                break
        chrom.append(chosen)
    return chrom


def random_chrom(prob: PlacementProblem) -> List[int]:
    if not prob.services or not prob.candidate_nodes:
        return []
    return [random.choice(prob.candidate_nodes) for _ in range(len(prob.services))]


def mutate_one_gene(chrom: List[int], prob: PlacementProblem) -> List[int]:
    if not chrom:
        return chrom
    i = random.randrange(len(chrom))
    options = [n for n in prob.candidate_nodes if n != chrom[i]]
    if options:
        chrom = list(chrom)
        chrom[i] = random.choice(options)
    return chrom


def to_allocation(chrom: List[int], prob: PlacementProblem):
    return [
        {"module_name": prob.services[i], "app": str(prob.service_to_app[i]), "id_resource": chrom[i]}
        for i in range(len(prob.services))
    ]


def _find_cloud_id(entities) -> int:
    for eid, edata in entities.items():
        if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
            return eid
    return max(entities.keys(), default=0)


def _get_app_source_node(app_id: str, users) -> int | None:
    for src in users.get("sources", []):
        if src.get("app") == str(app_id):
            return src.get("id_resource")
    return None


def _get_module_chain(app) -> List[str]:
    next_mod = {}
    first = None
    for msg in app.get("message", []):
        if msg.get("s") == "None":
            first = msg["d"]
        else:
            next_mod[msg["s"]] = msg["d"]
    if first is None:
        return [m["name"] for m in app.get("module", [])]
    chain = [first]
    while chain[-1] in next_mod:
        chain.append(next_mod[chain[-1]])
    return chain

