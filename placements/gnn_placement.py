"""
Graph Neural Network based Placement (GNNPlacement)
Uses GCN-inspired approach for service placement.
"""
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from placements.placement import Placement


class GNNPlacement(Placement):
    """
    GNN-based placement using Graph Convolutional Network approach:
    - Node embeddings: learned through message passing
    - Service-Device matching: computed via embedding similarity
    - Training: iterative feature propagation + supervised signal
    
    Simplified GNN without deep learning frameworks:
    - Manual implementation of GCN layers
    - Feature aggregation from neighbors
    - Service placement as node classification
    """
    
    def __init__(self, hidden_dim=32, num_layers=2, epochs=30, 
                 learning_rate=0.01, residual_alpha=0.3, seed=None):
        self.name = "GNNPlacement"
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.residual_alpha = residual_alpha  # new_embed = (1-alpha)*old + alpha*transformed
        self.seed = seed
        
        # Learnable parameters (simplified)
        self.W_layers = []
        self.device_base_features = {}  # Fixed base features (4 dims)
        self.device_embeddings = {}      # Learned embeddings (hidden_dim)
        self.service_base_features = {}  # Fixed base features (4 dims)
        self.service_embeddings = {}     # Learned embeddings (hidden_dim)
    
    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation using GNN.
        
        Returns:
            List of allocation dicts
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Build graph representation
        print("[GNN] Building graph representation...")
        graph_data = self._build_graph_representation(topology, applications, users)
        
        # Initialize embeddings
        print("[GNN] Initializing embeddings...")
        self._initialize_embeddings(graph_data)
        
        # Train GNN (simplified)
        print(f"[GNN] Training GNN for {self.epochs} epochs...")
        self._train_gnn(graph_data)
        
        # Generate allocation
        print("[GNN] Generating allocation from embeddings...")
        allocation = self._generate_allocation_from_embeddings(graph_data)
        
        return allocation
    
    def _build_graph_representation(self, topology, applications, users):
        """Build unified graph with devices and services."""
        # Device graph
        G_devices = nx.Graph()
        entities = {e["id"]: e for e in topology["entity"]}
        
        for entity in topology["entity"]:
            G_devices.add_node(entity["id"], 
                             node_type="device",
                             **entity)
        
        for link in topology["link"]:
            weight = link["PR"] + 2500000 / link["BW"]
            G_devices.add_edge(link["s"], link["d"], 
                             edge_type="network",
                             weight=weight, **link)
        
        # Find cloud (YAFS: type CLOUD; or model cloud for backward compat)
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
        
        # Service graph (from app dependencies)
        G_services = nx.DiGraph()
        services = []
        service_info = {}
        
        for app in applications:
            # Add service nodes
            for module in app["module"]:
                sname = module["name"]
                services.append(sname)
                G_services.add_node(sname,
                                  node_type="service",
                                  app=app["id"],
                                  ram=module.get("RAM", 1))
                service_info[sname] = {
                    "app": app["id"],
                    "ram": module.get("RAM", 1),
                    "name": sname
                }
            
            # Add service dependencies (from messages)
            for msg in app.get("message", []):
                if msg["s"] != "None":
                    G_services.add_edge(msg["s"], msg["d"],
                                      edge_type="dependency")
        
        # User gateways
        app_gw = defaultdict(list)
        for src in users.get("sources", []):
            gw = src["id_resource"]
            if gw not in app_gw[src["app"]]:
                app_gw[src["app"]].append(gw)
        
        # Precompute latencies
        devices = list(G_devices.nodes())
        latencies = self._precompute_latencies(G_devices, app_gw, devices)
        
        graph_data = {
            "G_devices": G_devices,
            "G_services": G_services,
            "entities": entities,
            "cloud_id": cloud_id,
            "services": services,
            "service_info": service_info,
            "devices": devices,
            "app_gw": dict(app_gw),
            "latencies": latencies
        }
        
        return graph_data
    
    def _precompute_latencies(self, G, app_gw, devices):
        """Precompute latencies."""
        latencies = {}
        all_gw = set()
        for gws in app_gw.values():
            all_gw.update(gws)
        
        for gw in all_gw:
            try:
                lengths = nx.single_source_dijkstra_path_length(G, gw, weight='weight')
            except:
                lengths = {}
            for node in devices:
                latencies[(gw, node)] = lengths.get(node, 1e9)
        
        return latencies
    
    def _initialize_embeddings(self, graph_data):
        """Initialize node embeddings with features."""
        # Device embeddings: [IPT_norm, RAM_norm, is_cloud, degree_norm]
        G_dev = graph_data["G_devices"]
        max_ipt = max(graph_data["entities"][d].get("IPT", 1) for d in graph_data["devices"])
        max_ram = max(graph_data["entities"][d].get("RAM", 1) for d in graph_data["devices"] 
                     if d != graph_data["cloud_id"])
        max_degree = max(dict(G_dev.degree()).values()) if G_dev.number_of_nodes() > 0 else 1
        
        for device in graph_data["devices"]:
            entity = graph_data["entities"][device]
            ipt_norm = entity.get("IPT", 1) / max(max_ipt, 1)
            ram_norm = entity.get("RAM", 1) / max(max_ram, 1) if device != graph_data["cloud_id"] else 1.0
            is_cloud = 1.0 if device == graph_data["cloud_id"] else 0.0
            degree_norm = G_dev.degree(device) / max(max_degree, 1)
            
            # Store base features separately (fixed)
            self.device_base_features[device] = np.array([ipt_norm, ram_norm, is_cloud, degree_norm])
            
            # Random initialization for learned embeddings
            self.device_embeddings[device] = np.random.randn(self.hidden_dim) * 0.01
        
        # Service embeddings: [ram_norm, app_norm, in_degree_norm, out_degree_norm]
        G_serv = graph_data["G_services"]
        max_ram_serv = max(info["ram"] for info in graph_data["service_info"].values())
        num_apps = max(info["app"] for info in graph_data["service_info"].values()) + 1
        max_in_deg = max(dict(G_serv.in_degree()).values()) if G_serv.number_of_nodes() > 0 else 1
        max_out_deg = max(dict(G_serv.out_degree()).values()) if G_serv.number_of_nodes() > 0 else 1
        
        for service in graph_data["services"]:
            info = graph_data["service_info"][service]
            ram_norm = info["ram"] / max(max_ram_serv, 1)
            app_norm = info["app"] / max(num_apps, 1)
            in_deg_norm = G_serv.in_degree(service) / max(max_in_deg, 1)
            out_deg_norm = G_serv.out_degree(service) / max(max_out_deg, 1)
            
            # Store base features separately (fixed)
            self.service_base_features[service] = np.array([ram_norm, app_norm, in_deg_norm, out_deg_norm])
            
            # Random initialization for learned embeddings
            self.service_embeddings[service] = np.random.randn(self.hidden_dim) * 0.01
        
        # Initialize weight matrices: first layer (4 + hidden_dim -> hidden_dim), rest (hidden_dim -> hidden_dim)
        input_dim = 4 + self.hidden_dim
        self.W_layers = []
        for layer_idx in range(self.num_layers):
            in_dim = input_dim if layer_idx == 0 else self.hidden_dim
            W = np.random.randn(in_dim, self.hidden_dim) * 0.01
            self.W_layers.append(W)
    
    def _aggregate_neighbors(self, graph_data, node, embeddings, graph):
        """Aggregate features from neighbors (simplified GCN)."""
        neighbors = list(graph.neighbors(node))
        if not neighbors:
            return embeddings[node]
        
        # Mean aggregation
        neighbor_embeds = [embeddings[n] for n in neighbors]
        agg = np.mean(neighbor_embeds, axis=0)
        
        # Combine with self
        combined = (embeddings[node] + agg) / 2.0
        return combined
    
    def _train_gnn(self, graph_data):
        """
        Train GNN: forward pass with residual updates, then simple weight update.
        Loss uses normalized latency so embedding similarity affects the score.
        """
        for epoch in range(self.epochs):
            # Forward pass with residual (prevents embedding collapse)
            new_device_embeds = {}
            for device in graph_data["devices"]:
                agg = self._aggregate_neighbors(
                    graph_data, device, self.device_embeddings, graph_data["G_devices"]
                )
                current = np.concatenate([self.device_base_features[device], agg])
                for layer_idx in range(self.num_layers):
                    current = np.tanh(np.dot(current, self.W_layers[layer_idx]))
                # Residual: blend old and new so embeddings evolve gradually
                new_device_embeds[device] = (
                    (1 - self.residual_alpha) * self.device_embeddings[device]
                    + self.residual_alpha * current
                )
            
            new_service_embeds = {}
            for service in graph_data["services"]:
                agg = self._aggregate_neighbors(
                    graph_data, service, self.service_embeddings, graph_data["G_services"]
                )
                current = np.concatenate([self.service_base_features[service], agg])
                for layer_idx in range(self.num_layers):
                    current = np.tanh(np.dot(current, self.W_layers[layer_idx]))
                new_service_embeds[service] = (
                    (1 - self.residual_alpha) * self.service_embeddings[service]
                    + self.residual_alpha * current
                )
            
            self.device_embeddings = new_device_embeds
            self.service_embeddings = new_service_embeds
            
            total_loss = self._compute_placement_loss(graph_data)
            
            # Simple weight update: try perturbed W, re-forward to get new embeddings, keep if loss improves
            layer_idx = epoch % self.num_layers
            W_old = self.W_layers[layer_idx].copy()
            self.W_layers[layer_idx] += np.random.randn(*W_old.shape).astype(np.float64) * self.learning_rate
            # Re-run forward pass with perturbed W to get candidate embeddings
            cand_device = {}
            for device in graph_data["devices"]:
                agg = self._aggregate_neighbors(
                    graph_data, device, self.device_embeddings, graph_data["G_devices"]
                )
                current = np.concatenate([self.device_base_features[device], agg])
                for li in range(self.num_layers):
                    current = np.tanh(np.dot(current, self.W_layers[li]))
                cand_device[device] = (1 - self.residual_alpha) * self.device_embeddings[device] + self.residual_alpha * current
            cand_service = {}
            for service in graph_data["services"]:
                agg = self._aggregate_neighbors(
                    graph_data, service, self.service_embeddings, graph_data["G_services"]
                )
                current = np.concatenate([self.service_base_features[service], agg])
                for li in range(self.num_layers):
                    current = np.tanh(np.dot(current, self.W_layers[li]))
                cand_service[service] = (1 - self.residual_alpha) * self.service_embeddings[service] + self.residual_alpha * current
            # Temporarily use candidate embeddings to compute loss
            old_dev, old_serv = self.device_embeddings, self.service_embeddings
            self.device_embeddings, self.service_embeddings = cand_device, cand_service
            loss_after = self._compute_placement_loss(graph_data)
            self.device_embeddings, self.service_embeddings = old_dev, old_serv
            if loss_after < total_loss:
                total_loss = loss_after
                self.device_embeddings, self.service_embeddings = cand_device, cand_service
                # keep perturbed W (already in self.W_layers[layer_idx])
            else:
                self.W_layers[layer_idx] = W_old
            
            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                print(f"[GNN] Epoch {epoch+1}/{self.epochs}, Loss: {total_loss:.4f}")
    
    def _compute_placement_loss(self, graph_data):
        """
        Compute placement quality loss (simplified).
        Lower loss = better. Uses normalized latency so embedding similarity matters.
        """
        # Normalize latencies to [0, 1] so similarity (in [-1,1]) can affect the score
        all_lats = []
        for service in graph_data["services"]:
            info = graph_data["service_info"][service]
            gws = graph_data["app_gw"].get(info["app"], [1])
            for device in graph_data["devices"]:
                avg_lat = np.mean([graph_data["latencies"].get((gw, device), 1e9) for gw in gws])
                all_lats.append(avg_lat)
        min_lat = min(all_lats)
        max_lat = max(all_lats)
        lat_span = max(max_lat - min_lat, 1.0)
        
        total_loss = 0.0
        for service in graph_data["services"]:
            service_embed = self.service_embeddings[service]
            info = graph_data["service_info"][service]
            app_id = info["app"]
            gws = graph_data["app_gw"].get(app_id, [1])
            
            best_score = -1e9
            for device in graph_data["devices"]:
                device_embed = self.device_embeddings[device]
                similarity = np.dot(service_embed, device_embed) / (
                    np.linalg.norm(service_embed) * np.linalg.norm(device_embed) + 1e-9
                )
                avg_lat = np.mean([graph_data["latencies"].get((gw, device), 1e9) for gw in gws])
                lat_norm = (avg_lat - min_lat) / lat_span  # in [0, 1], lower is better
                lat_penalty = -lat_norm  # so score = similarity - lat_norm
                score = similarity + lat_penalty
                best_score = max(best_score, score)
            
            total_loss -= best_score
        
        return total_loss
    
    def _generate_allocation_from_embeddings(self, graph_data):
        """Generate allocation using learned embeddings."""
        allocation = []
        device_usage = defaultdict(float)
        
        for service in graph_data["services"]:
            service_embed = self.service_embeddings[service]
            info = graph_data["service_info"][service]
            app_id = info["app"]
            gws = graph_data["app_gw"].get(app_id, [1])
            
            # Score each device
            scores = {}
            for device in graph_data["devices"]:
                device_embed = self.device_embeddings[device]
                
                # Embedding similarity
                similarity = np.dot(service_embed, device_embed) / (
                    np.linalg.norm(service_embed) * np.linalg.norm(device_embed) + 1e-9
                )
                
                # Latency component
                avg_lat = np.mean([graph_data["latencies"].get((gw, device), 1e9) for gw in gws])
                lat_score = -avg_lat / 1000.0
                
                # Capacity component
                cap = graph_data["entities"][device].get("RAM", 1e9)
                usage = device_usage[device]
                cap_score = 0 if usage + info["ram"] <= cap else -10.0
                
                # Cloud bonus (fallback)
                cloud_bonus = 5.0 if device == graph_data["cloud_id"] else 0.0
                
                total_score = similarity + lat_score + cap_score + cloud_bonus
                scores[device] = total_score
            
            # Select best device
            best_device = max(scores, key=scores.get)
            device_usage[best_device] += info["ram"]
            
            allocation.append({
                "module_name": service,
                "app": str(app_id),
                "id_resource": best_device
            })
        
        return allocation
