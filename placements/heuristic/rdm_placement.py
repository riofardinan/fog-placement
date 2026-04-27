"""
Random Placement (RDMPlacement) — Pakpahan et al. (2025) Algorithm 3.

FindSuitableNode:
1. Randomly select a node; check if it has sufficient RAM.
2. Repeat up to MAX_ATTEMPTS times.
3. If no suitable fog node found, place on cloud.
"""
import random
from placements.placement import Placement


class RDMPlacement(Placement):
    MAX_ATTEMPTS = 100

    def __init__(self):
        super().__init__()
        self.name = "RDMPlacement"

    def generate_allocation(self, topology, applications, users):
        entities = {e["id"]: e for e in topology["entity"]}
        cloud_id = self._find_cloud_id(entities)
        fog_nodes = [e["id"] for e in topology["entity"] if e["id"] != cloud_id]

        caps = {n: float(entities[n].get("RAM", 0)) for n in fog_nodes}
        allocation = []

        for app in applications:
            app_id = str(app["id"])
            for module in app.get("module", []):
                ram_req = float(module.get("RAM", 1))
                node = self._find_suitable_node(fog_nodes, caps, ram_req, cloud_id)
                if node != cloud_id:
                    caps[node] -= ram_req
                allocation.append({
                    "module_name": module["name"],
                    "app": app_id,
                    "id_resource": node,
                })

        return allocation

    def _find_suitable_node(self, fog_nodes, caps, ram_req, cloud_id):
        """Algorithm 3: random selection with max_attempts, fallback to cloud."""
        for _ in range(self.MAX_ATTEMPTS):
            node = random.choice(fog_nodes)
            if caps.get(node, 0) >= ram_req:
                return node
        return cloud_id