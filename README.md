# Fog Computing Microservice Placement Simulation

Riset fog computing dengan fokus pada placement algorithms menggunakan YAFS 3.1 (Yet Another Fog Simulator).

## Struktur Proyek

```
FOG/
├── config/                              # Parameter konfigurasi
│   ├── topology_params.py              # Topologi BA graph, resource fog/cloud, PR, BW, IPT
│   ├── app_params.py                   # Parameter microservice chain (modul, RAM, instruksi, pesan)
│   └── users_params.py                 # Parameter user (lambda request, gateway assignment)
│
├── generator/                           # STEP 1: Generate skenario
│   ├── generate_scenario.py            # Hasilkan topology, aplikasi, users → JSON
│   └── generate_placements.py          # Hasilkan alokasi per algoritma → JSON
│
├── scenarios/                           # Output generator (JSON + Excel analisis)
│
├── placements/                          # Implementasi algoritma placement (tanpa YAFS)
│   ├── placement.py                    # Base class + helper BFS/adjacency
│   ├── rdm_placement.py                # Random (RDM) — Algorithm 3
│   ├── sm_placement.py                 # Sort and Match (SM) — Algorithm 4
│   ├── ffha_placement.py               # FirstFitHopAware (FFHA) — Algorithm 4.3
│   ├── hop2_placement.py               # Hop2 — Algorithm 4.2
│   ├── hop3_placement.py               # Hop3 — Algorithm 4.4
│   ├── fff_placement.py                # FrameworkFirstFit (FFF) — Algorithm 5
│   ├── ga_placement.py                 # Genetic Algorithm (GA) — Algorithm 6
│   ├── ilp_placement.py                # Integer Linear Programming (ILP)
│   ├── gr_placement.py                 # Greedy (GR)
│   ├── cn_placement.py                 # Complex Network (CN)
│   ├── pso_placement.py                # Particle Swarm Optimization (PSO)
│   ├── cngapso_placement.py            # Hybrid CN+GA/PSO
│   ├── gnn_placement.py                # (eksperimental) GNN-based
│   └── rl_placement.py                 # (eksperimental) RL-based
│
├── runner/                              # STEP 2: Jalankan simulasi YAFS
│   ├── run_experiment.py               # Eksperimen multi-instance (14 × 10 runs) ← UTAMA
│   ├── run_simulation.py               # Jalankan satu skenario (single placement)
│   ├── run_all_placements.py           # Jalankan semua placement satu skenario
│   ├── path_routing.py                 # Factory routing strategy (DeviceSpeedAwareRouting, dll)
│   └── json_population.py              # JSONPopulation wrapper untuk YAFS
│
├── results/                             # Output simulasi
│   └── apps_{N}/run_{R}/{ALGO}/        # sim_trace.csv, sim_trace_link.csv per run
│
├── analysis/                            # STEP 3: Analisis dan visualisasi
│   ├── plot_mean_response_time_sweep.py # Plot Fig. 8 + Fig. 9 gaya paper (sweep apps)
│   ├── placement_static.py             # Analisis statik skenario sebelum simulasi
│   ├── analyze_results.py              # Analisis runtime single-skenario
│   └── *.png                           # Output grafik
│
├── requirements.txt
└── README.md
```

## Prerequisites

- **Python 3.12** (direkomendasikan untuk YAFS 3.1)
- **YAFS 3.1** diinstall dari source (lihat Setup)

---

## Setup

### 1. Buat virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Clone dan patch YAFS 3.1

```bash
git clone --branch YAFS3.1 https://github.com/acsicuib/YAFS
cd YAFS
pip install -e .
cd ../FOG
```

**Patch wajib** — nonaktifkan link-queue serialization di YAFS 3.1 agar setara dengan YAFS 0.3.0 (versi yang dipakai paper). Edit `YAFS/src/yafs/core.py`, cari bagian `shift_time` dan ubah menjadi:

```python
# No link queue serialization — matches YAFS 0.3.0 behaviour (paper)
shift_time = 0.0
```

### 3. Install dependensi Python

```bash
pip install -r requirements.txt
```

---

## Workflow

Terdapat **dua workflow**: eksperimen lengkap multi-instance dan single-run.

### A. Eksperimen Lengkap

Jalankan satu perintah berikut untuk menjalankan seluruh 14 instance × 10 runs × 7 algoritma:

```bash
python runner/run_experiment.py
```

Script ini secara otomatis:

1. Generate topologi sekali (fixed seed) dan bagikan ke semua run
2. Generate aplikasi + users per `(num_apps, run)` untuk variabilitas
3. Generate alokasi placement per algoritma
4. Jalankan simulasi YAFS dan simpan trace

Output disimpan di:

```
results/
└── apps_5/
│   ├── run_1/
│   │   ├── RDM/sim_trace.csv
│   │   ├── SM/sim_trace.csv
│   │   ├── FFHA/sim_trace.csv
│   │   ├── Hop2/sim_trace.csv
│   │   ├── Hop3/sim_trace.csv
│   │   ├── FFF/sim_trace.csv
│   │   └── GA/sim_trace.csv
│   └── run_2/ ...
├── apps_10/ ...
└── apps_70/ ...
```

### B. Analisis dan Visualisasi (Fig. 8 dan Fig. 9 Paper)

```bash
python analysis/plot_mean_response_time_sweep.py
```

Menghasilkan dua grafik di folder `analysis/`:

- `**mean_response_time_by_apps.png**` — Fig. 8 gaya paper: mean response time vs jumlah aplikasi, satu garis per algoritma
- `**response_components_by_algo.png**` — Fig. 9 gaya paper: stacked bar Service Time / Latency / Wait Time per algoritma

### C. Single-Run (Eksplorasi / Debugging)

#### Step 1 — Generate skenario

```bash
python generator/generate_scenario.py
python generator/generate_placements.py
```

Output di `scenarios/`: `networkDefinition.json`, `appDefinition.json`, `usersDefinition.json`, `allocDefinition*.json`

#### Step 2 — Jalankan simulasi

```bash
# Satu algoritma
python runner/run_simulation.py --placement RDMPlacement --duration 10000

# Semua algoritma
python runner/run_all_placements.py
```

Parameter `--routing` yang tersedia:


| Nilai              | Kelas YAFS                 | Keterangan                                           |
| ------------------ | -------------------------- | ---------------------------------------------------- |
| `device_speed`     | `DeviceSpeedAwareRouting`  | **Default** — shortest path hop count (setara paper) |
| `weighted_latency` | `SelectionWeightedLatency` | Bobot PR + size/BW                                   |
| `load_aware`       | `LoadAwareRouting`         | Gabungan latency + load node                         |


#### Step 3 — Analisis hasil single-run

```bash
# Analisis statik (sebelum simulasi)
python -m analysis.placement_static

# Analisis runtime (setelah simulasi)
python analysis/analyze_results.py
```

---

## Output Simulasi

Setiap direktori `{ALGO}/` berisi:

### `sim_trace.csv`

Kolom utama YAFS per event:


| Kolom            | Keterangan                       |
| ---------------- | -------------------------------- |
| `id`             | ID request                       |
| `time_emit`      | Waktu request dikirim user       |
| `time_reception` | Waktu tiba di node tujuan        |
| `time_in`        | Mulai diproses (setelah antri)   |
| `time_out`       | Selesai diproses                 |
| `service`        | Waktu CPU = `time_out - time_in` |


**Rumus response time** (per request `id`):

```
Response Time = max(time_out) - min(time_emit)
Service Time  = sum(time_out - time_in)
Latency       = sum(time_reception - time_emit)
Wait Time     = sum(time_in - time_reception)
```

### `sim_trace_link.csv`

Metrics per network link (latency, message size, buffer utilization).

---

## Algoritma Placement


| Kode   | Nama              | Deskripsi Singkat                                       |
| ------ | ----------------- | ------------------------------------------------------- |
| `RDM`  | Random            | Pilih node random, coba MAX_ATTEMPTS=100 kali           |
| `SM`   | Sort and Match    | Sort node by IPT desc, first-fit RAM                    |
| `FFHA` | FirstFitHopAware  | Pack ke node yang sama, BFS hop berikutnya jika penuh   |
| `Hop2` | Hop2              | Modul pertama di gateway, berikutnya node berbeda (BFS) |
| `Hop3` | Hop3              | Modul pertama 1 hop dari gateway, berikutnya BFS        |
| `FFF`  | FrameworkFirstFit | FFHA dengan adjacency hierarkis (FG→FOG→CFG→Cloud)      |
| `GA`   | Genetic Algorithm | Optimasi hop chain; pop=100, gen=250, mut=0.3           |


---

## Catatan Perbedaan dengan Paper


| Aspek         | Paper (YAFS 0.3.0)    | Simulasi ini (YAFS 3.1)                   |
| ------------- | --------------------- | ----------------------------------------- |
| Link queue    | Tidak ada serialisasi | Dinonaktifkan (`shift_time=0`)            |
| Topology seed | Tidak dipublikasikan  | `TOPOLOGY_SEED=42` di `run_experiment.py` |
| Lambda seed   | Tidak dipublikasikan  | Seeded per `run % 100`                    |
| IPT           | 1.500–3.000 (Table 1) | 100–1.000 (dikonfirmasi dari Fig. 9)      |


Karena seed topologi dan lambda tidak diketahui, beberapa algoritma (SM, Hop2, Hop3) masih menunjukkan selisih ~10–30% dari nilai paper. FFHA, FFF, dan GA sudah dalam toleransi <2%.

---

## Referensi

```
Pakpahan et al. (2025).
Comparative analysis of rule-based heuristic algorithms for microservice chain placement in fog computing.

Isaac Lera, Carlos Guerrero and Carlos Juiz.
YAFS: A simulator for IoT scenarios in fog computing.
IEEE Access. Vol. 7(1), pages 91745–91758, 2019.
DOI: 10.1109/ACCESS.2019.2927895
```

- [YAFS GitHub](https://github.com/acsicuib/YAFS)

