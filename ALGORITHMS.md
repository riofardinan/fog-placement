# Placement Algorithms

Dokumentasi lengkap untuk 5 placement algorithms (implementasi berdasarkan Lera et al., IEEE IoT Journal 2019). Semua penjelasan teknis, parameter, dan referensi ada di dokumen ini.

## 1. CNPlacement (Complex Network)

### Konsep

Menggunakan teori Complex Network untuk mengidentifikasi struktur komunitas dalam topologi fog, lalu menempatkan service berdasarkan proximity dan fitness.

### Algorithm Flow

```
1. Girvan-Newman Community Detection
   - Iteratively remove edges with highest betweenness
   - Generate hierarchical communities
   - Sort by depth (deeper = smaller, more cohesive)

2. For each app (sorted by deadline):
   a. Build app DAG from service dependencies
   b. Compute transitive closure partitions
   c. For each user gateway:
      - Find community containing gateway
      - Order devices by fitness
      - Place services in community
   d. Fallback to cloud if no feasible placement

3. Device fitness = exec_time + network_time
   where:
     exec_time = total_MIPS / IPT(device)
     network_time = shortest_path(gateway, device)
```

### Model Matematis

- Network weight: `w(e) = PR(e) + msg_size / BW(e)`
- Exec time: `total_MIPS / IPT(device)`
- Net time: shortest path dengan weight
- Fitness: `f(d) = exec_time(d) + net_time(gateway, d)`
- Apps diurutkan by deadline (prioritas)

### Kelebihan

- Memanfaatkan struktur topologi
- Locality-aware placement
- Consideration untuk app dependencies

### Kekurangan

- Computational overhead untuk large networks
- Tidak adaptif terhadap workload changes

---

## 2. GAPlacement (Genetic Algorithm)

### Konsep

Evolutionary optimization yang evolusikan populasi placement untuk minimize fitness (latency + util + risk).

### Algorithm Flow

```
1. Initialization
   - Build Top-K candidate devices per service
   - Random population of chromosomes
   - Repair capacity violations

2. Evolution (for G generations):
   a. Selection: tournament (pick best of k random)
   b. Crossover: one-point between parents
   c. Mutation: reassign to candidate devices
   d. Repair: move services if overload
   e. Elitism: keep best individual

3. Return best chromosome as allocation
```

### Chromosome & Fitness

```
Chromosome: [dev_0, dev_1, ..., dev_n]
            ↓      ↓           ↓
         service_0, 1, ..., n

Fitness = α * (avg_latency / 1000)
        + β * (Σ max(0, (demand_i - cap_i)/cap_i) / num_devices)
        + γ * (avg_risk / 1.0)

Default: α=0.4, β=0.3, γ=0.3

util_penalty = Σ_d max(0, (demand_d - capacity_d) / capacity_d)
avg_risk     = 1 - availability (simplified)
```

### Parameter Default

- Population: 40, Generations: 60
- Crossover rate: 0.9, Mutation rate: 0.2
- Tournament k: 3, Candidate k: 10 (Top-K devices per service)

### Kelebihan

- Global search (tidak stuck di local optima)
- Multi-objective optimization
- Parallel exploration

### Kekurangan

- Convergence time (banyak generations)
- Parameter tuning needed

---

## 3. ILPPlacement (Integer Linear Programming)

### Konsep

Mathematical optimization dengan binary variables dan linear constraints.

### Formulation

```
Variables:
  x[(g,s),d] ∈ {0,1}
  = 1 if service s from gateway g placed on device d

Objective:
  minimize Σ x[(g,s),d] * latency(g, d)

Constraints:
  1. Assignment: Σ_d x[(g,s),d] = 1  ∀(g,s)
     (each user-service assigned exactly once)
  
  2. Capacity: Σ_(g,s) x[(g,s),d] * RAM(s) ≤ CAP(d)  ∀d
     (device capacity not exceeded)

Solver: CBC (via PuLP)
Strategy: solve per app (sorted by deadline)
```

### Kelebihan

- Optimal solution (for single objective)
- Mathematically provable
- Handles constraints explicitly

### Kekurangan

- Scalability (NP-hard for large instances)
- Single objective (latency only)
- Requires solver (PuLP/CBC)

---

## 4. RLPlacement (Reinforcement Learning)

### Konsep

Q-learning agent yang belajar policy optimal untuk sequential service placement.

### MDP Formulation

```
State: (service_idx, capacity_bins)
  capacity_bins = discretized [low<5, med<15, high] per device

Action: select device ∈ {all devices}

Reward:
  r = -latency/1000 
    - 10 * capacity_violation
    - 0.1 * cost
  
  where capacity_violation = max(0, demand - capacity)
        cost = 10 (cloud) or 1 (fog)

Q-update:
  Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
```

### Training

```
For E episodes:
  1. ε-greedy action selection
     - Explore (prob ε): random device
     - Exploit (prob 1-ε): argmax_a Q(s,a)
  
  2. Take action, observe reward
  
  3. Update Q-value (temporal difference)
  
  4. Decay ε (exploration rate)

Inference: greedy policy (argmax_a Q(s,a))
```

### Parameter Default

- Episodes: 50
- Learning rate (α): 0.1, Discount factor (γ): 0.95
- Exploration rate (ε): 0.3 → 0.01 (decay 0.95)
- State discretization: capacity bins (`low<5`, `medium<15`, `high`)

### Kelebihan

- Learns from experience
- Adaptable (retrain with new data)
- No need for explicit model

### Kekurangan

- Training time (many episodes)
- State space explosion (large systems)
- Q-table size grows

---

## 5. GNNPlacement (Graph Neural Network)

### Konsep

Graph Convolutional Network yang belajar node embeddings untuk match services dengan devices.

### Architecture

```
Input Graphs:
  G_devices: physical topology
  G_services: app dependency DAG

Node Features:
  Device: [IPT_norm, RAM_norm, is_cloud, degree_norm]
  Service: [RAM_norm, app_norm, in_deg, out_deg]

GCN Layers (L layers):
  h^(l+1)_v = tanh(W^(l) · AGG({h^(l)_u : u ∈ N(v)}))
  
  where AGG = mean pooling

Placement Score:
  score(s,d) = cos_sim(embed_s, embed_d) 
             - latency/1000 
             - 10*capacity_violation
             + 5*is_cloud
```

### Training

```
For T epochs:
  1. Message passing (aggregate neighbors)
  2. Feature transformation (W · agg)
  3. Compute placement loss
     loss = -Σ max_d score(s,d)
  4. Update embeddings (simplified gradient)

Inference:
  For each service:
    - Score all devices
    - Select argmax_d score(s,d)
    - Respect capacity
```

### Parameter Default

- Hidden dimension: 32, GCN layers: 2
- Training epochs: 30, Learning rate: 0.01 (simplified gradient)

### Kelebihan

- Captures graph structure
- Learns feature representations
- Scalable (with proper batching)

### Kekurangan

- Training complexity
- Requires graph data
- Black box (hard to interpret)

---

## Comparison Matrix


| Algorithm | Time Complexity | Optimality | Scalability | Adaptability |
| --------- | --------------- | ---------- | ----------- | ------------ |
| CN        | O(n³)           | Heuristic  | Medium      | Low          |
| GA        | O(P·G·n·m)      | Near-opt   | Good        | Medium       |
| ILP       | Exponential     | Optimal    | Poor        | Low          |
| RL        | O(E·n²)         | Near-opt   | Good        | High         |
| GNN       | O(T·L·          | E          | )           | Heuristic    |


Where:

- n = #services, m = #devices
- P = population, G = generations (GA)
- E = episodes (RL)
- T = epochs, L = layers (GNN)

---

## When to Use Which?

### CNPlacement

**Use when:**

- Network has clear community structure
- Topology is relatively static
- Need locality-aware placement

### GAPlacement

**Use when:**

- Multi-objective optimization needed
- Want balance between exploration/exploitation
- Have time for evolution

### ILPPlacement

**Use when:**

- Need provably optimal solution
- Problem size is manageable (<100 services)
- Single objective (latency) is primary

### RLPlacement

**Use when:**

- Environment dynamics expected
- Can afford training phase
- Want adaptive policy

### GNNPlacement

**Use when:**

- Graph structure is important
- Have graph data available
- Want learned representations
- Need scalability

---

## References

1. Isaac Lera, Carlos Guerrero, Carlos Juiz. "Availability-aware Service Placement Policy in Fog Computing Based on Graph Partitions", *IEEE Internet of Things Journal*, 2019.

