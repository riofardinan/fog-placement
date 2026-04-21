"""
Complex Network based Placement (CNPlacement)
Aligned with baseline CNoptimization.py (Lera et al., IEEE IoT Journal 2019).
- Girvan-Newman community detection on FOG graph only (no cloud node).
- Per-app link weight: PR + (app source message bytes) / BW.
- Place for every client (gateway); same module can be on multiple devices.
- Device fitness: exec_time + net_time (exec_time = app total MIPS / IPT).
- Cloud replica added at end (paper assumption).
"""
import itertools
import operator
import networkx as nx
from networkx.algorithms import community
from placements.placement import Placement


class CNPlacement(Placement):
    """
    CN-based placement (baseline-aligned):
    - Graph: fog nodes only (no cloud in G).
    - weightNetwork(appId): set edge weight = PR + size/BW per app source message.
    - For each app, for each client: find community containing client, place; accumulate.
    - Fitness: total_mips / IPT + shortest_path_length.
    - Add cloud replica per service at end.
    """

    def __init__(self):
        super().__init__()
        self.name = "CNPlacement"
        self.sorted_communities = []
        self.node_busy_resources = {}

    def generate_allocation(self, topology, applications, users):
        entities = {e["id"]: e for e in topology["entity"]}
        cloud_id = None
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
        if cloud_id is None:
            cloud_id = max(entities.keys(), default=0) + 1

        # Build FOG-ONLY graph (no cloud) and fog-only links for re-weighting
        fog_ids = [e["id"] for e in topology["entity"] if e["id"] != cloud_id]
        G = nx.Graph()
        for nid in fog_ids:
            G.add_node(nid, **entities[nid])
        fog_links = []
        for link in topology["link"]:
            s, d = link["s"], link["d"]
            if s != cloud_id and d != cloud_id:
                G.add_edge(s, d, PR=link["PR"], BW=link["BW"])
                fog_links.append(link)

        self.node_busy_resources = {n: 0.0 for n in G.nodes()}
        self._compute_communities(G)

        apps_closures = {}
        app_source_bytes = {}
        app_total_mips = {}
        for app in applications:
            app_graph = self._build_app_graph(app)
            apps_closures[app["id"]] = None
            if app_graph.number_of_nodes() > 0:
                source_nodes = [n for n in app_graph.nodes() if app_graph.in_degree(n) == 0]
                if source_nodes:
                    apps_closures[app["id"]] = self._transitive_closure_partition(
                        source_nodes[0], app_graph
                    )
            # Per-app source message size and total MIPS (for fitness)
            total_mips = 0
            src_bytes = 3000000
            for msg in app.get("message", []):
                total_mips += msg.get("instructions", 40000)
                if msg.get("s") == "None":
                    src_bytes = msg.get("bytes", 3000000)
            app_source_bytes[app["id"]] = src_bytes
            app_total_mips[app["id"]] = total_mips

        app_gateways = self._get_app_gateways(users)
        allocation = []
        sorted_apps = sorted(applications, key=lambda a: a.get("MaxLatency", 10000))

        for app in sorted_apps:
            app_id = app["id"]
            if apps_closures[app_id] is None:
                for module in app["module"]:
                    allocation.append({
                        "module_name": module["name"],
                        "app": str(app_id),
                        "id_resource": cloud_id,
                    })
                continue

            # Per-app network weighting (baseline: weightNetwork(appId))
            size = float(app_source_bytes[app_id])
            for link in fog_links:
                u, v = link["s"], link["d"]
                if G.has_edge(u, v):
                    G[u][v]["weight"] = float(link["PR"]) + size / float(link["BW"])

            gateways = app_gateways.get(app_id, [1])
            for gateway in gateways:
                if gateway not in G:
                    continue
                placed = False
                for comm, depth in self.sorted_communities:
                    if gateway in comm:
                        placement = self._place_app_in_community(
                            app, gateway, comm, apps_closures[app_id],
                            entities, G, app_total_mips[app_id],
                        )
                        if placement:
                            for mod_name, device_id in placement.items():
                                allocation.append({
                                    "module_name": mod_name,
                                    "app": str(app_id),
                                    "id_resource": device_id,
                                })
                            placed = True
                            break
                if not placed:
                    for module in app["module"]:
                        allocation.append({
                            "module_name": module["name"],
                            "app": str(app_id),
                            "id_resource": cloud_id,
                        })

        # Cloud replica per service (paper assumption)
        for app in applications:
            for module in app["module"]:
                allocation.append({
                    "module_name": module["name"],
                    "app": str(app["id"]),
                    "id_resource": cloud_id,
                })

        return allocation

    def _compute_communities(self, G):
        communities_gen = community.girvan_newman(G)
        communities = {frozenset(G.nodes()): 0}
        level = 1
        for comms in itertools.islice(communities_gen, G.number_of_nodes()):
            for c in comms:
                communities[frozenset(c)] = level
            level += 1
        self.sorted_communities = sorted(
            communities.items(), key=lambda x: x[1], reverse=True
        )

    def _transitive_closure_partition(self, source, app_graph):
        closures = {}
        def dfs(node, level):
            closures.setdefault(level, set())
            desc = set(nx.descendants(app_graph, node)) | {node}
            fs = frozenset(desc)
            if fs not in closures[level]:
                closures[level].add(fs)
                for n in app_graph.neighbors(node):
                    closures.setdefault(level + 1, set()).add(frozenset([node]))
                    dfs(n, level + 1)
        dfs(source, 0)
        prev = closures[0]
        for lvl in sorted(closures.keys()):
            current = set().union(*closures[lvl])
            extra = {s for s in prev if len(s & current) == 0}
            closures[lvl] |= extra
            prev = closures[lvl]
        return closures

    def _place_app_in_community(self, app, gateway, community, closures, entities, G, total_mips):
        remaining = {m["name"] for m in app["module"]}
        placement = {}
        ordered_devices = self._order_devices_by_fitness(
            community, gateway, entities, G, total_mips
        )
        for dev in ordered_devices:
            if dev not in entities:
                continue
            free_res = entities[dev].get("RAM", 0) - self.node_busy_resources.get(dev, 0)
            for lvl in sorted(closures.keys()):
                for sset in sorted(closures[lvl], key=len, reverse=True):
                    sset_names = {
                        next((m["name"] for m in app["module"] if m["id"] == sid), f"{app['id']}_{sid}")
                        for sid in sset
                    }
                    if sset_names & remaining:
                        req = sum(m["RAM"] for m in app["module"] if m["name"] in sset_names)
                        if free_res >= req:
                            for sname in sset_names:
                                if sname in remaining:
                                    placement[sname] = dev
                            remaining -= sset_names
                            free_res -= req
                            self.node_busy_resources[dev] = self.node_busy_resources.get(dev, 0) + req
                            if not remaining:
                                return placement
        return placement if not remaining else None

    def _order_devices_by_fitness(self, community, gateway, entities, G, total_mips):
        """Fitness = exec_time + net_time (baseline: total_mips/IPT + path length)."""
        fitness = {}
        for dev in community:
            if dev not in entities:
                continue
            ipt = entities[dev].get("IPT", 100)
            exec_time = total_mips / max(ipt, 1)
            try:
                net_time = nx.shortest_path_length(G, source=gateway, target=dev, weight="weight")
            except Exception:
                net_time = 1e9
            fitness[dev] = exec_time + net_time
        return [d for d, _ in sorted(fitness.items(), key=operator.itemgetter(1))]

    def _build_app_graph(self, app):
        G = nx.DiGraph()
        for module in app["module"]:
            G.add_node(module["id"])
        for msg in app.get("message", []):
            if msg.get("s") != "None":
                src_id = next((m["id"] for m in app["module"] if m["name"] == msg["s"]), None)
                dst_id = next((m["id"] for m in app["module"] if m["name"] == msg["d"]), None)
                if src_id is not None and dst_id is not None:
                    G.add_edge(src_id, dst_id)
        return G

    def _get_app_gateways(self, users):
        app_gw = {}
        for src in users.get("sources", []):
            app_id = src["app"]
            gw = src["id_resource"]
            if app_id not in app_gw:
                app_gw[app_id] = []
            if gw not in app_gw[app_id]:
                app_gw[app_id].append(gw)
        return app_gw
