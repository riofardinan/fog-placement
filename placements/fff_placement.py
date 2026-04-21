"""
FrameworkFirstFit Placement (FFFPlacement) — Pakpahan et al. (2025), Section 4.5.

Inspired by FogFrame [Skarlat & Schulte, 2021].
Uses a HIERARCHICAL adjacency dict (modified Algorithm 5) to enforce:
  Gateway (FG)  → Fog nodes (FOG) + Fog heads (CFG)
  Fog nodes     → Fog nodes (FOG) + Fog heads (CFG)
  Fog heads     → Fog heads (CFG) + Fog nodes (FOG) + Cloud
  Cloud         → Fog heads (CFG) only

Placement logic: same as FirstFitHopAware (packing allowed, BFS through
hierarchical adjacency). The hierarchical structure ensures modules are
placed gateway → fog/cfghead → cloud in that natural order.

Note: FG can reach both FOG AND CFG neighbors. In a Barabasi-Albert
graph, FG gateways are often directly connected to CFG hub nodes; excluding
CFG from FG's adj would cause immediate cloud fallback with empty BFS.
"""
from placements.placement import Placement


class FFFPlacement(Placement):

    def __init__(self):
        super().__init__()
        self.name = "FFFPlacement"

    def generate_allocation(self, topology, applications, users):
        entities = {e["id"]: e for e in topology["entity"]}
        cloud_id = self._find_cloud_id(entities)
        adj = self._build_hierarchical_adjacency(topology, entities, cloud_id)
        fog_nodes = [e["id"] for e in topology["entity"] if e["id"] != cloud_id]
        caps = {n: float(entities[n].get("RAM", 0)) for n in fog_nodes}

        allocation = []

        for app in applications:
            app_id = str(app["id"])
            chain = self._get_module_chain(app)
            module_ram = {m["name"]: float(m.get("RAM", 1)) for m in app.get("module", [])}
            source = self._get_app_source_node(app["id"], users) or fog_nodes[0]

            # Start from the source gateway, follow hierarchical adjacency (packing allowed)
            current_node = source
            cloud_used = False

            for mod_name in chain:
                ram_req = module_ram.get(mod_name, 1.0)

                if cloud_used:
                    node = cloud_id
                elif caps.get(current_node, 0) >= ram_req:
                    # Pack on current node
                    node = current_node
                    caps[current_node] -= ram_req
                else:
                    # BFS through hierarchical adjacency
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

    def _build_hierarchical_adjacency(self, topology, entities, cloud_id):
        """
        Build hierarchical adjacency dict (modified Algorithm 5).
        Rules:
          FG  → actual neighbors that are FOG or CFG
          FOG → actual neighbors that are FOG or CFG
          CFG → actual neighbors that are FOG or CFG (+ cloud)
          CLOUD → all CFG nodes
        """
        node_type = {e["id"]: e.get("type", "FOG") for e in topology["entity"]}
        net_adj = self._build_adjacency_map(topology)
        adj = {}

        for e in topology["entity"]:
            nid = e["id"]
            ntype = node_type[nid]
            neighbors = net_adj.get(nid, [])

            if ntype == "CLOUD":
                adj[nid] = [nb for nb in neighbors if node_type.get(nb) == "CFG"]
            elif ntype == "CFG":
                adj[nid] = [nb for nb in neighbors
                            if node_type.get(nb) in ("FOG", "CFG")]
                if cloud_id not in adj[nid]:
                    adj[nid].append(cloud_id)
            elif ntype in ("FOG", "FG"):
                # Both FOG and FG can reach FOG or CFG neighbors (upward hierarchy)
                adj[nid] = [nb for nb in neighbors
                            if node_type.get(nb) in ("FOG", "CFG")]

        return adj
