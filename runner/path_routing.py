"""
Custom YAFS routing strategies for this repository.

These classes are adapted from the experimental `exp` project so that
this repo does not depend on the `exp` folder.
"""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

import networkx as nx
from yafs.selection import Selection
from yafs.path_routing import DeviceSpeedAwareRouting as YAFSDeviceSpeedAwareRouting


class SelectionWeightedLatency(Selection):
    """
    Selects the destination with the lowest weighted shortest-path latency.

    The path cost is computed using the edge attribute ``weight`` in the
    YAFS topology graph.
    """

    def __init__(self) -> None:
        self.cache: Dict[Tuple[Hashable, Tuple[Hashable, ...]], Tuple[List[Hashable], Hashable]] = {}
        self.invalid_cache_value: bool = True
        self.controlServices: Dict[Tuple[Hashable, str], Tuple[List[Hashable], Hashable]] = {}
        super(SelectionWeightedLatency, self).__init__()

    def compute_BEST_DES(self, node_src, alloc_DES, sim, DES_dst, message):
        try:
            bestCost = float("inf")
            bestPath: List[Hashable] = []
            bestDES = None

            for dev in DES_dst:
                node_dst = alloc_DES[dev]

                cost = nx.shortest_path_length(
                    sim.topology.G,
                    source=node_src,
                    target=node_dst,
                    weight="weight",
                )

                if cost < bestCost:
                    bestCost = cost
                    bestPath = list(
                        nx.shortest_path(
                            sim.topology.G,
                            source=node_src,
                            target=node_dst,
                            weight="weight",
                        )
                    )
                    bestDES = dev

            return bestPath, bestDES

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return [], None

    def get_path(
        self,
        sim,
        app_name,
        message,
        topology_src,
        alloc_DES,
        alloc_module,
        traffic,
        from_des,
    ):
        node_src = topology_src
        service = message.dst
        DES_dst = alloc_module[app_name][message.dst]

        if self.invalid_cache_value:
            self.invalid_cache_value = False
            self.cache = {}

        key = (node_src, tuple(DES_dst))

        if key not in self.cache:
            self.cache[key] = self.compute_BEST_DES(
                node_src, alloc_DES, sim, DES_dst, message
            )

        path, des = self.cache[key]
        self.controlServices[(node_src, service)] = (path, des)

        return [path], [des]

    def get_path_from_failure(
        self,
        sim,
        message,
        link,
        alloc_DES,
        alloc_module,
        traffic,
        ctime,
        from_des,
    ):
        idx = message.path.index(link[0])

        if idx == len(message.path):
            return [], []
        else:
            node_src = message.path[idx]

            path, des = self.get_path(
                sim,
                message.app_name,
                message,
                node_src,
                alloc_DES,
                alloc_module,
                traffic,
                from_des,
            )

            if len(path[0]) > 0:
                concPath = message.path[0 : message.path.index(path[0][0])] + path[0]
                message.dst_int = node_src
                return [concPath], des
            else:
                return [], []


class LoadAwareRouting(Selection):
    """
    Routing strategy that balances path latency with node load.

    The cost combines shortest-path latency (edge weight) and a simple
    estimate of node utilization based on how many DES are assigned to
    each node and its processing capacity.
    """

    def __init__(self, load_weight: float = 0.5, cooldown_period: float = 0.05):
        super(LoadAwareRouting, self).__init__()
        self.load_weight = load_weight
        # Duration (simulation time units) for which a previous decision is reused
        self.cooldown_period = cooldown_period

        # Cached shortest-path latencies between nodes
        self.latency_cache: Dict[Tuple[Hashable, Hashable], float] = {}
        # Last routing decision per (source node, service)
        self.last_decision: Dict[Tuple[Hashable, str], Tuple[List[List[Hashable]], List[Hashable]]] = {}
        # Last time a decision was taken for (source node, service)
        self.last_time: Dict[Tuple[Hashable, str], float] = {}
        self.controlServices: Dict[Tuple[Hashable, str], Tuple[List[Hashable], Hashable]] = {}

    def get_latency(self, sim, src, dst):
        """Cache Dijkstra latency so it is not recomputed for every message."""
        key = (src, dst)
        if key not in self.latency_cache:
            try:
                self.latency_cache[key] = nx.shortest_path_length(
                    sim.topology.G,
                    source=src,
                    target=dst,
                    weight="weight",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                self.latency_cache[key] = float("inf")
        return self.latency_cache[key]

    def compute_BEST_DES(self, node_src, alloc_DES, sim, DES_dst, node_loads):
        bestCost = float("inf")
        bestDES = None

        for dev in DES_dst:
            node_dst = alloc_DES[dev]
            latency = self.get_latency(sim, node_src, node_dst)

            if latency == float("inf"):
                continue

            active_des = node_loads.get(node_dst, 0)
            node_capacity = sim.topology.G.nodes[node_dst].get("IPT", 1.0)

            utilization = active_des / max(node_capacity, 1e-9)

            cost = latency + self.load_weight * (utilization**1.2)

            if cost < bestCost:
                bestCost = cost
                bestDES = dev

        if bestDES is not None:
            node_dst = alloc_DES[bestDES]
            bestPath = list(
                nx.shortest_path(
                    sim.topology.G,
                    source=node_src,
                    target=node_dst,
                    weight="weight",
                )
            )
            return bestPath, bestDES
        return [], None

    def get_path(
        self,
        sim,
        app_name,
        message,
        topology_src,
        alloc_DES,
        alloc_module,
        traffic,
        from_des,
    ):
        node_src = topology_src
        service = message.dst
        cache_key = (node_src, service)
        time_now = sim.env.now

        # 1. Decision cooldown: if still within cooldown, reuse last decision
        if cache_key in self.last_time:
            if (time_now - self.last_time[cache_key]) < self.cooldown_period:
                return self.last_decision[cache_key]

        # 2. Recompute node loads only when cooldown has expired
        DES_dst = alloc_module[app_name][message.dst]
        node_loads: Dict[Hashable, int] = {}
        for n in alloc_DES.values():
            node_loads[n] = node_loads.get(n, 0) + 1

        # 3. Compute best path for current loads
        path, des = self.compute_BEST_DES(node_src, alloc_DES, sim, DES_dst, node_loads)

        # 4. Cache decision for cooldown window
        res = [path], [des]
        self.last_decision[cache_key] = res
        self.last_time[cache_key] = time_now
        self.controlServices[(node_src, service)] = (path, des)

        return res


def create_routing_strategy(name: str):
    """
    Factory for routing strategies used in this repository.

    Parameters
    ----------
    name:
        One of:
        - ``"device_speed"``  (default, matches paper — nx.shortest_path unweighted/hop-count)
        - ``"weighted_latency"`` (custom `SelectionWeightedLatency`, equivalent to device_speed
          when edges have no 'weight' attribute)
        - ``"load_aware"`` (custom `LoadAwareRouting`)
    """
    normalized = name.lower()
    mapping = {
        "device_speed": YAFSDeviceSpeedAwareRouting,
        "device_speed_aware": YAFSDeviceSpeedAwareRouting,  # backward alias
        "weighted_latency": SelectionWeightedLatency,
        "selection_weighted_latency": SelectionWeightedLatency,
        "load_aware": LoadAwareRouting,
        "loadaware": LoadAwareRouting,
    }

    if normalized not in mapping:
        valid = {"device_speed", "weighted_latency", "load_aware"}
        raise ValueError(
            f"Unknown routing strategy '{name}'. "
            f"Valid options are: {sorted(valid)}"
        )

    return mapping[normalized]()

