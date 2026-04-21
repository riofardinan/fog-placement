"""
Hop3 Placement (Hop3Placement) — Pakpahan et al. (2025), Section 4.4.

Variant of Hop2. Key difference:
- First module → placed 1 hop AWAY from source (not ON source node).
- Subsequent modules: same as Hop2 (BFS from previous module's neighbors).
Each module is placed on a DIFFERENT node (no packing).
Fallback: cloud. Once cloud used, all remaining modules go to cloud.
"""
from placements.placement import Placement


class Hop3Placement(Placement):

    def __init__(self):
        super().__init__()
        self.name = "Hop3Placement"

    def generate_allocation(self, topology, applications, users):
        entities = {e["id"]: e for e in topology["entity"]}
        cloud_id = self._find_cloud_id(entities)
        adj = self._build_adjacency_map(topology)
        fog_nodes = [e["id"] for e in topology["entity"] if e["id"] != cloud_id]
        caps = {n: float(entities[n].get("RAM", 0)) for n in fog_nodes}

        allocation = []

        for app in applications:
            app_id = str(app["id"])
            chain = self._get_module_chain(app)
            module_ram = {m["name"]: float(m.get("RAM", 1)) for m in app.get("module", [])}
            source = self._get_app_source_node(app["id"], users) or fog_nodes[0]

            current_node = source
            cloud_used = False

            for i, mod_name in enumerate(chain):
                ram_req = module_ram.get(mod_name, 1.0)

                if cloud_used:
                    node = cloud_id
                elif i == 0:
                    # First module: 1 hop AWAY from source (Hop3 = 1-hop from source)
                    found = self._bfs_find_next_node(source, adj, caps, ram_req, cloud_id)
                    if found is not None:
                        node = found
                        caps[found] -= ram_req
                        current_node = found
                    else:
                        node = cloud_id
                        cloud_used = True
                else:
                    # Subsequent modules: different node, BFS from neighbors
                    found = self._bfs_find_next_node(current_node, adj, caps, ram_req, cloud_id)
                    if found is not None:
                        node = found
                        caps[found] -= ram_req
                        current_node = found
                    else:
                        node = cloud_id
                        cloud_used = True

                allocation.append({
                    "module_name": mod_name,
                    "app": app_id,
                    "id_resource": node,
                })

        return allocation
