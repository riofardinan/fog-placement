"""
Reinforcement Learning based Placement (RLPlacement)
Uses Q-learning for service placement optimization.
"""
import random
import numpy as np
import networkx as nx
from collections import defaultdict
from placements.placement import Placement

class RLPlacement(Placement):
    """
    RL-based placement using Q-learning:
    - State: (service_id, available_resources_vector)
    - Action: select device for service
    - Reward: -latency (baseline-style: network delay + execution delay)
    - Policy: ε-greedy with decaying exploration
    
    Training phases:
    1. Exploration: random placements to learn Q-values
    2. Exploitation: use learned Q-values for optimal placement
    """
    
    def __init__(self, episodes=200, alpha=0.1, gamma=0.9, epsilon=0.2,
                 epsilon_decay=1.0, epsilon_min=0.2, seed=None):
        super().__init__()
        self.name = "RLPlacement"
        self.episodes = episodes
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor (baseline 0.9)
        self.epsilon = epsilon  # Exploration rate (baseline 0.2 fixed)
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.seed = seed
        
        # Q-table: Q[(state_hash, action)] = value
        self.Q = defaultdict(float)
        
        # Reward weights (baseline: latency-only objective)
        self.reward_weights = {
            "latency": -1.0,
            "capacity_violation": 0.0,
            "cost": 0.0,
        }
    
    def generate_allocation(self, topology, applications, users):
        """
        Generate allocation using Q-learning.
        
        Returns:
            List of allocation dicts
        """
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        # Build environment
        env = self._build_environment(topology, applications, users)
        
        # Train Q-learning agent
        print(f"[RL] Training Q-learning agent for {self.episodes} episodes...")
        self._train_q_learning(env)
        
        # Generate final allocation using learned policy
        print("[RL] Generating allocation with learned policy...")
        allocation = self._generate_final_allocation(env)
        
        return allocation
    
    def _build_environment(self, topology, applications, users):
        """Build RL environment."""
        G = nx.Graph()
        entities = {e["id"]: e for e in topology["entity"]}
        
        for entity in topology["entity"]:
            G.add_node(entity["id"], **entity)
        
        for link in topology["link"]:
            # Baseline network weighting: PR + size/BW with size = 3_000_000 bytes
            weight = link["PR"] + 3000000 / link["BW"]
            G.add_edge(link["s"], link["d"], weight=weight, **link)
        
        # Find cloud (YAFS: type CLOUD; or model cloud for backward compat)
        cloud_id = 0
        for eid, edata in entities.items():
            if edata.get("type") == "CLOUD" or edata.get("model") == "cloud":
                cloud_id = eid
                break
        
        # Services list
        services = []
        service_info = {}
        for app in applications:
            for module in app["module"]:
                sid = len(services)
                sname = module["name"]
                services.append(sname)
                service_info[sname] = {
                    "id": sid,
                    "app": str(app["id"]),
                    "ram": module.get("RAM", 1),
                    "name": sname
                }
        
        # Device list
        devices = list(G.nodes())
        device_info = {d: entities[d] for d in devices}
        
        # User gateways per app
        app_gw = defaultdict(list)
        for src in users.get("sources", []):
            gw = src["id_resource"]
            if gw not in app_gw[src["app"]]:
                app_gw[src["app"]].append(gw)
        
        # Precompute latencies
        latencies = self._precompute_latencies(G, app_gw, devices)
        # Baseline: reward = -latency with SOURCE SERVICE only per (app, gateway)
        app_source_sid = {}
        for app in applications:
            source_name = None
            for msg in app.get("message", []):
                if msg.get("s") == "None":
                    source_name = msg.get("d")
                    break
            if source_name is None and app.get("module"):
                source_name = app["module"][0]["name"]
            if source_name is None:
                continue
            for sid, sname in enumerate(services):
                if sname == source_name and service_info.get(sname, {}).get("app") == str(app["id"]):
                    app_source_sid[str(app["id"])] = sid
                    break
        
        env = {
            "G": G,
            "entities": entities,
            "cloud_id": cloud_id,
            "services": services,
            "service_info": service_info,
            "devices": devices,
            "device_info": device_info,
            "app_gw": dict(app_gw),
            "latencies": latencies,
            "app_source_sid": app_source_sid,
        }
        
        return env
    
    def _precompute_latencies(self, G, app_gw, devices):
        """Precompute shortest path latencies."""
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
    
    def _get_state_hash(self, service_idx, device_capacities):
        """
        Create state representation.
        State = (service_idx, discretized_capacity_bins)
        """
        # Discretize capacities into bins (low, medium, high)
        capacity_bins = []
        for cap in device_capacities:
            if cap < 5:
                capacity_bins.append(0)  # low
            elif cap < 15:
                capacity_bins.append(1)  # medium
            else:
                capacity_bins.append(2)  # high
        
        return (service_idx, tuple(capacity_bins))
    
    def _compute_latency_source_only(self, env, placement_by_sid):
        """Baseline: avg over (app, gateway) of net_delay(gw, device[source_sid]) + 40000/IPT."""
        latencies = env["latencies"]
        app_gw = env["app_gw"]
        app_source_sid = env.get("app_source_sid", {})
        device_info = env["device_info"]
        lat_vals = []
        for app_id, gws in app_gw.items():
            sid = app_source_sid.get(app_id) or app_source_sid.get(str(app_id))
            if sid is None or sid >= len(placement_by_sid):
                continue
            dev = placement_by_sid[sid]
            ipt = device_info.get(dev, {}).get("IPT", 100)
            exec_delay = 40000.0 / max(float(ipt), 1.0)
            for gw in gws:
                net_delay = latencies.get((gw, dev), 1e9)
                lat_vals.append(net_delay + exec_delay)
        return (sum(lat_vals) / len(lat_vals)) if lat_vals else 1e9

    def _get_reward(self, env, placement, service_name, device):
        """
        Baseline: reward = -latency where latency is from FULL placement
        using SOURCE SERVICE only (per app, gateway).
        """
        placement_by_sid = [
            placement.get(sname, device if sname == service_name else env["cloud_id"])
            for sname in env["services"]
        ]
        avg_lat = self._compute_latency_source_only(env, placement_by_sid)
        return -avg_lat
    
    def _train_q_learning(self, env):
        """Train Q-learning agent."""
        epsilon = self.epsilon
        
        for episode in range(self.episodes):
            # Initialize placement
            placement = {}
            device_capacities = [env["device_info"][d].get("RAM", 1e9) 
                                for d in env["devices"]]
            
            episode_reward = 0
            
            # Place each service sequentially
            for service_idx, service_name in enumerate(env["services"]):
                state = self._get_state_hash(service_idx, device_capacities)
                
                # ε-greedy action selection
                if random.random() < epsilon:
                    # Explore: random device
                    action = random.choice(env["devices"])
                else:
                    # Exploit: best Q-value
                    q_values = {device: self.Q[(state, device)] 
                               for device in env["devices"]}
                    action = max(q_values, key=q_values.get)
                
                # Take action
                placement[service_name] = action
                reward = self._get_reward(env, placement, service_name, action)
                episode_reward += reward
                
                # Update device capacity
                device_idx = env["devices"].index(action)
                device_capacities[device_idx] -= env["service_info"][service_name]["ram"]
                
                # Next state
                if service_idx + 1 < len(env["services"]):
                    next_state = self._get_state_hash(service_idx + 1, device_capacities)
                    
                    # Q-learning update
                    max_next_q = max(self.Q[(next_state, device)] 
                                    for device in env["devices"])
                    
                    self.Q[(state, action)] += self.alpha * (
                        reward + self.gamma * max_next_q - self.Q[(state, action)]
                    )
                else:
                    # Terminal state
                    self.Q[(state, action)] += self.alpha * (
                        reward - self.Q[(state, action)]
                    )
            
            # Decay epsilon
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            
            if (episode + 1) % max(1, self.episodes // 10) == 0:
                print(f"[RL] Episode {episode+1}/{self.episodes}, "
                      f"Reward: {episode_reward:.2f}, Epsilon: {epsilon:.3f}")
    
    def _generate_final_allocation(self, env):
        """Generate final allocation using learned Q-values (greedy)."""
        allocation = []
        placement = {}
        device_capacities = [env["device_info"][d].get("RAM", 1e9) 
                            for d in env["devices"]]
        
        for service_idx, service_name in enumerate(env["services"]):
            state = self._get_state_hash(service_idx, device_capacities)
            
            # Select best action based on Q-values
            q_values = {device: self.Q.get((state, device), 0.0) 
                       for device in env["devices"]}
            best_device = max(q_values, key=q_values.get)
            
            # Capacity check (fallback to cloud if needed)
            device_cap = env["device_info"][best_device].get("RAM", 1e9)
            device_usage = sum(env["service_info"][s]["ram"] 
                             for s, d in placement.items() if d == best_device)
            device_usage += env["service_info"][service_name]["ram"]
            
            if device_usage > device_cap and best_device != env["cloud_id"]:
                best_device = env["cloud_id"]
            
            placement[service_name] = best_device
            
            # Update capacity
            device_idx = env["devices"].index(best_device)
            device_capacities[device_idx] -= env["service_info"][service_name]["ram"]
            
            allocation.append({
                "module_name": service_name,
                "app": str(env["service_info"][service_name]["app"]),
                "id_resource": best_device
            })
        
        # Baseline: mandatory cloud replica per service
        for service_name in env["services"]:
            allocation.append({
                "module_name": service_name,
                "app": str(env["service_info"][service_name]["app"]),
                "id_resource": env["cloud_id"]
            })
        
        return allocation
