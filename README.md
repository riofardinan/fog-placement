# Fog Computing Placement Simulation

Riset fog computing dengan fokus pada placement algorithms menggunakan YAFS 3.1 (Yet Another Fog Simulator).

## Struktur Proyek

```
FOG/
├── config/                          # Konfigurasi parameter
│   ├── topology_params.py          # Parameter topologi (cloud, fog, network)
│   ├── app_params.py               # Parameter aplikasi dan services
│   └── users_params.py             # Parameter users/IoT devices
│
├── generator/                       # RUN 1: Generate scenario
│   ├── generate_scenario.py        # Generate topology, app, users → JSON
│   └── generate_placements.py      # Generate allocation per placement → JSON
│
├── scenarios/                       # Output generator (JSON files)
│
├── placements/                      # Implementasi placement algorithms
│   ├── __init__.py
│   ├── cn_placement.py             # Complex Network based
│   ├── ga_placement.py             # Genetic Algorithm based
│   ├── ilp_placement.py            # Integer Linear Programming based
│   ├── rl_placement.py             # Reinforcement Learning based
│   └── gnn_placement.py            # Graph Neural Network based
│
├── runner/                          # RUN 2: Execute simulation
│   ├── run_simulation.py           # Run single placement
│   ├── run_all_placements.py       # Run all placements
│   └── json_population.py          # JSON population for YAFS
│
├── results/                         # Output simulasi (per placement)
│
├── analysis/                        # RUN 3: Analisis dan perbandingan hasil
│   ├── analyze_results.py          # Bandingkan metrics (latency, energy, utilization, dll)
│   ├── comparison_results.csv      # Output tabel perbandingan
│   └── *.png                       # Plot perbandingan (latency, energy, load balance, dll)
│
├── requirements.txt                 # Dependencies
├── CONFIGURATION.md                 # Detail konfigurasi
└── ALGORITHMS.md                    # Detail algoritma placement
```

## Prerequisites

- **Python 3.12** (direkomendasikan untuk YAFS 3.1)
- **uv**

## Setup

### 1. Install YAFS 3.1

```bash
# Clone YAFS
git clone --branch YAFS3.1 https://github.com/acsicuib/YAFS
cd YAFS

# Install dengan uv (package manager YAFS 3.1)
uv sync
uv pip install -e .

# Kembali ke project FOG
cd ../FOG
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Workflow

### RUN 1: Generate Scenario

#### Step 1: Generate Topology, Applications, Users

```bash
python generator/generate_scenario.py
```

Output:

- `scenarios/networkDefinition.json`
- `scenarios/appDefinition.json`
- `scenarios/usersDefinition.json`

#### Step 2: Generate Placement Allocations

```bash
python generator/generate_placements.py
```

Output:

- `scenarios/allocDefinitionCN.json`
- `scenarios/allocDefinitionGA.json`
- `scenarios/allocDefinitionILP.json`
- `scenarios/allocDefinitionRL.json`
- `scenarios/allocDefinitionGNN.json`

### RUN 2: Execute Simulation

#### Run Single Placement

```bash
# CNPlacement
python runner/run_simulation.py --placement CNPlacement --duration 20000

# GAPlacement
python runner/run_simulation.py --placement GAPlacement --duration 20000

# ILPPlacement
python runner/run_simulation.py --placement ILPPlacement --duration 20000

# RLPlacement
python runner/run_simulation.py --placement RLPlacement --duration 20000

# GNNPlacement
python runner/run_simulation.py --placement GNNPlacement --duration 20000
```

#### Run All Placements

```bash
python runner/run_all_placements.py
```

### RUN 3: Analyze Results

```bash
python analysis/analyze_results.py
```

Output:

- `analysis/comparison_results.csv`: Tabel perbandingan semua placement.
- `analysis/*.png`: Plot perbandingan (latency, energy, execution time, load balance, nodes utilization).

Metrics: 

- Service latency 
- Energy consumption 
- Resource utilization  
- Execution time.

## Output Results

Setiap simulasi menghasilkan 2 file CSV:

### `sim_trace.csv`

Metrics per module:

- Service time
- Latency (time_emit - time_in)
- Module allocation (TOPO.dst)
- Application ID

### `sim_trace_link.csv`

Metrics per network link:

- Latency
- Message size
- Buffer utilization

## Citation

```
Isaac Lera, Carlos Guerrero and Carlos Juiz. 
YAFS: A simulator for IoT scenarios in fog computing. 
IEEE Access. Vol. 7(1), pages 91745-91758, 2019.
DOI: 10.1109/ACCESS.2019.2927895
```

## References

- [YAFS GitHub](https://github.com/acsicuib/YAFS)

