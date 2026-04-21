"""
Sort and Match Placement (SMPlacement) — Pakpahan et al. (2025) Algorithm 4.

FindSuitableNode:
1. Sort fog nodes by IPT descending (highest processing speed first).
2. Iterate sorted list; place on first node with sufficient RAM.
3. If no fog node fits, place on cloud.
"""
from placements.placement import Placement


class SMPlacement(Placement):

    def __init__(self):
        super().__init__()
        self.name = "SMPlacement"

    def generate_allocation(self, topology, applications, users):
        entities = {e["id"]: e for e in topology["entity"]}
        cloud_id = self._find_cloud_id(entities)

        # Sort fog nodes by IPT descending (Algorithm 4)
        sorted_fog = sorted(
            [e for e in topology["entity"] if e["id"] != cloud_id],
            key=lambda e: e.get("IPT", 0),
            reverse=True,
        )

        caps = {e["id"]: float(e.get("RAM", 0)) for e in sorted_fog}
        allocation = []

        for app in applications:
            app_id = str(app["id"])
            for module in app.get("module", []):
                ram_req = float(module.get("RAM", 1))
                node = self._find_suitable_node(sorted_fog, caps, ram_req, cloud_id)
                if node != cloud_id:
                    caps[node] -= ram_req
                allocation.append({
                    "module_name": module["name"],
                    "app": app_id,
                    "id_resource": node,
                })

        return allocation

    def _find_suitable_node(self, sorted_fog, caps, ram_req, cloud_id):
        """Algorithm 4: first fog node (by IPT desc) with sufficient RAM."""
        for e in sorted_fog:
            if caps.get(e["id"], 0) >= ram_req:
                return e["id"]
        return cloud_id
