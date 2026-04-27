"""
FirstFitHopAware Placement (FFHAPlacement) — Pakpahan et al. (2025), Section 4.3.

Uses Algorithm 5 (createMAP) to build direct-neighbor adjacency dict.

FindSuitableNode per module in chain:
1. Start from the source node (gateway) for the first module.
2. Try current_node (same node as previous module = packing).
3. If doesn't fit → BFS from current_node through adjacency dict.
4. If BFS finds nothing → place on cloud, mark cloud_used.

First module of each app starts from the SOURCE NODE (gateway), consistent
with Table 5 of the paper which shows first-hop ≈ 0.08–0.80 (near gateway).
"""
from placements.placement import Placement


class FFHAPlacement(Placement):

    def __init__(self):
        super().__init__()
        self.name = "FFHAPlacement"

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

            # Start from the source gateway (same as Hop2), packing allowed
            current_node = source
            cloud_used = False

            for mod_name in chain:
                ram_req = module_ram.get(mod_name, 1.0)

                if cloud_used:
                    node = cloud_id
                elif caps.get(current_node, 0) >= ram_req:
                    # Pack on current node (source for first module, last-placed for rest)
                    node = current_node
                    caps[current_node] -= ram_req
                else:
                    # BFS from current_node for next available fog node
                    found = self._bfs_find_node(current_node, adj, caps, ram_req, cloud_id)
                    if found is not None:
                        node = found
                        current_node = found
                        caps[current_node] -= ram_req
                    else:
                        node = cloud_id
                        cloud_used = True

                allocation.append({
                    "module_name": mod_name,
                    "app": app_id,
                    "id_resource": node,
                })

        return allocation
