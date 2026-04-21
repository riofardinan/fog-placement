"""
Greedy-based Placement (GRPlacement)

Greedy "Closest Resource First" (CRF) placement:
- Objective: minimize latency by always choosing the nearest feasible device
  (shortest network delay) that has enough RAM for the service.
- Fallback: if no fog device can host the service, place it on the cloud.

This is a placement-style adaptation of `GRoptimization.py` from the baseline
implementation, refactored to use:
- Topology/app/users dicts from JSON (FOG project style)
- The common link latency model: PR + (3_000_000 / BW)
"""

import operator
import networkx as nx

from placements.placement import Placement


class GRPlacement(Placement):
    """
    Greedy placement:
    - For each application (sorted by deadline / MaxLatency)
    - For each gateway (client) of that app
    - For each service/module in the app
      - Sort devices by shortest-path latency from the client
      - Pick the first device with enough free RAM
      - If none, fallback to cloud
    """

    def __init__(self):
        super().__init__()
        self.name = "GRPlacement"
        self.network_distances = {}

    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation using Greedy Closest-Resource-First strategy.

        Args:
            topology: dict with "entity" and "link"
            applications: list of app dicts
            users: dict with "sources"

        Returns:
            List[dict]: allocation entries:
                {"module_name": str, "app": str, "id_resource": int}
        """
        # Build graph and entity map
        G = self._build_graph(topology)
        entities = {e["id"]: e for e in topology["entity"]}

        # Find cloud node
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break

        # Device capacities (mutable RAM)
        device_caps = {n: entities[n].get("RAM", 1e9) for n in G.nodes()}
        # Unlimited cloud capacity (if present)
        if cloud_id in device_caps:
            device_caps[cloud_id] = 1e18

        # Gateways per app
        app_gateways = self._get_app_gateways(users)

        # Pre-compute network distances (per gateway, per device)
        all_gateways = set()
        for gws in app_gateways.values():
            all_gateways.update(gws)
        self._compute_network_distances(G, all_gateways)

        allocation = []

        # Sort applications by deadline / MaxLatency (lower first)
        sorted_apps = sorted(applications, key=lambda a: a.get("MaxLatency", 10000))

        # Greedy placement per app
        for app in sorted_apps:
            app_id = app["id"]
            modules = list(app["module"])
            gateways = app_gateways.get(app_id, [1])

            for gw in gateways:
                for module in modules:
                    sname = module["name"]
                    required_ram = module.get("RAM", 1)

                    # Order devices by increasing latency from this gateway
                    nodes_by_distance = sorted(
                        G.nodes(),
                        key=lambda d: self.network_distances.get((gw, d), float("inf")),
                    )

                    placed = False
                    for dev_id in nodes_by_distance:
                        # Skip cloud here; handle it explicitly as fallback
                        if dev_id == cloud_id:
                            continue

                        free_ram = device_caps.get(dev_id, 0)
                        if free_ram >= required_ram:
                            allocation.append(
                                {
                                    "module_name": sname,
                                    "app": str(app_id),
                                    "id_resource": dev_id,
                                }
                            )
                            device_caps[dev_id] = free_ram - required_ram
                            placed = True
                            break

                    if not placed:
                        # Fallback to cloud
                        allocation.append(
                            {
                                "module_name": sname,
                                "app": str(app_id),
                                "id_resource": cloud_id,
                            }
                        )
                        # Cloud capacity is effectively unlimited; we don't decrease it

        # Baseline: mandatory cloud replica per service (paper assumption)
        for app in applications:
            for module in app["module"]:
                allocation.append({
                    "module_name": module["name"],
                    "app": str(app["id"]),
                    "id_resource": cloud_id,
                })

        return allocation

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_graph(self, topology):
        """Build NetworkX graph with baseline latency weights."""
        G = nx.Graph()
        for entity in topology["entity"]:
            G.add_node(entity["id"], **entity)
        for link in topology["link"]:
            # Baseline network weighting: PR + size/BW with size = 3_000_000 bytes
            weight = link["PR"] + 3000000 / link["BW"]
            G.add_edge(link["s"], link["d"], weight=weight, **link)
        return G

    def _compute_network_distances(self, G, gateways):
        """Compute shortest-path distances from each gateway to all devices."""
        self.network_distances = {}
        for gw in gateways:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, gw, weight="weight")
            except Exception:
                lengths = {}
            for node in G.nodes():
                self.network_distances[(gw, node)] = lengths.get(node, float("inf"))

    def _get_app_gateways(self, users):
        """Extract gateways per app from users (same style as other placements)."""
        app_gw = {}
        for src in users.get("sources", []):
            app_id = src["app"]
            gw = src["id_resource"]
            if app_id not in app_gw:
                app_gw[app_id] = []
            if gw not in app_gw[app_id]:
                app_gw[app_id].append(gw)
        return app_gw

