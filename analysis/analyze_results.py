"""
Result Analysis for YAFS Fog Computing Simulation.
Supports four metrics using YAFS Stats and Metrics:
- Service Latency (YAFS time_response / time_total_response)
- Energy Consumption (YAFS Stats.get_watt)
- Resource Utilization (YAFS time_service per node / total_time)
- Execution Time (YAFS time_service)

Visualizations: topology (layered), user distribution per app, and metric plots.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch

# Project root for YAFS and scenarios
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from yafs.topology import Topology
from yafs.stats import Stats
from yafs.metrics import Metrics

# Layer order (top to bottom): CLOUD, CFG, FOG, FG
_LAYER_ORDER = ("CLOUD", "CFG", "FOG", "FG")
# Energy model seperti exp/analyze_results.py: E = (service_time_ms/1000) * power_watt → Joule
POWER_FOG = 5.0
POWER_CLOUD = 100.0
_LAYER_COLORS = {
    "CLOUD": "#2d5a27",
    "CFG": "#e67e22",
    "FOG": "#5dade2",
    "FG": "#82e0aa",
}


def load_topology_data(scenarios_dir=None):
    """Load raw topology (entity + link) from networkDefinition.json for visualization."""
    if scenarios_dir is None:
        scenarios_dir = _project_root / "scenarios"
    net_file = scenarios_dir / "networkDefinition.json"
    if not net_file.exists():
        return None
    with open(net_file) as f:
        return json.load(f)


def load_users_data(scenarios_dir=None):
    """Load users (sources) from usersDefinition.json."""
    if scenarios_dir is None:
        scenarios_dir = _project_root / "scenarios"
    users_file = scenarios_dir / "usersDefinition.json"
    if not users_file.exists():
        return None
    with open(users_file) as f:
        return json.load(f)


def load_topology(scenarios_dir=None):
    """Load topology from networkDefinition.json for YAFS Stats.get_watt (WATT, model)."""
    if scenarios_dir is None:
        scenarios_dir = _project_root / "scenarios"
    net_file = scenarios_dir / "networkDefinition.json"
    if not net_file.exists():
        return None
    with open(net_file) as f:
        data = json.load(f)
    t = Topology()
    t.load(data)
    return t


def load_app_deadlines(scenarios_dir=None):
    """Load app deadlines from appDefinition.json."""
    if scenarios_dir is None:
        scenarios_dir = _project_root / "scenarios"
    app_file = scenarios_dir / "appDefinition.json"
    deadlines = {}
    if not app_file.exists():
        return deadlines
    with open(app_file, "r") as f:
        data = json.load(f)
        for app in data:
            try:
                app_id = int(app.get("id"))
                deadlines[app_id] = float(app.get("deadline", app.get("MaxLatency", 0)))
            except (TypeError, ValueError):
                continue
    return deadlines


def load_cloud_id(scenarios_dir=None):
    """Infer cloud node id from networkDefinition.json (fallback to 100)."""
    if scenarios_dir is None:
        scenarios_dir = _project_root / "scenarios"
    net_file = scenarios_dir / "networkDefinition.json"
    default_cloud_id = 100
    if not net_file.exists():
        return default_cloud_id
    try:
        with open(net_file, "r") as f:
            data = json.load(f)
        for ent in data.get("entity", []):
            if ent.get("type") == "CLOUD":
                return int(ent.get("id"))
    except Exception:
        pass
    return default_cloud_id


def load_planned_placement(placement_name, scenarios_dir=None, cloud_id=100):
    """
    Load planned placement per module from allocDefinition*.json.

    Uses same semantics as exp/analyze_results.py:
    - Prefer fog placement over cloud if both exist.
    """
    if scenarios_dir is None:
        scenarios_dir = _project_root / "scenarios"
    alloc_name = placement_name.replace("Placement", "")
    alloc_file = Path(scenarios_dir) / f"allocDefinition{alloc_name}.json"
    planned = {}
    if not alloc_file.exists():
        return planned
    try:
        with open(alloc_file, "r") as f:
            data = json.load(f)
    except Exception:
        return planned

    for item in data.get("initialAllocation", []):
        mod = item.get("module_name")
        try:
            res = int(item.get("id_resource"))
        except (TypeError, ValueError):
            continue
        if mod not in planned:
            planned[mod] = res
        elif planned[mod] == cloud_id and res != cloud_id:
            # prefer non-cloud if we have both
            planned[mod] = res
    return planned


def load_results(placement_name):
    """Load simulation results for a placement algorithm. Returns (df_trace, df_link, result_path)."""
    results_dir = _project_root / "results" / placement_name
    trace_file = results_dir / "sim_trace.csv"
    link_file = results_dir / "sim_trace_link.csv"
    if not trace_file.exists() or not link_file.exists():
        print(f"Warning: Results not found for {placement_name}")
        return None, None, None
    df_trace = pd.read_csv(trace_file)
    df_link = pd.read_csv(link_file)
    result_path = results_dir / "sim_trace"  # base path for YAFS Stats (no .csv)
    return df_trace, df_link, result_path


def analyze_placement(
    placement_name,
    result_path,
    topology,
    total_time=None,
    deadlines=None,
    planned_placement=None,
    cloud_id=None,
):
    """
    Analyze results using YAFS Stats.
    Returns metrics for: Service Latency, Energy Consumption, Resource Utilization, Execution Time.
    """
    link_path = Path(str(result_path) + "_link.csv")
    if not result_path or not result_path.with_suffix(".csv").exists() or not link_path.exists():
        return None
    path_str = str(result_path)
    try:
        stats = Stats(defaultPath=path_str)
    except Exception as e:
        print(f"Warning: Could not load YAFS Stats for {placement_name}: {e}")
        return None

    if total_time is None or total_time <= 0:
        total_time = float(stats.df["time_out"].max()) if len(stats.df) else 1.0

    stats.compute_times_df()
    df = stats.df

    # --- Time metrics in YAFS (service & total response) ---
    avg_service_latency = float(df["time_response"].mean())
    avg_total_response = float(df["time_total_response"].mean())
    max_service_latency = float(df["time_response"].max())

    # --- Derive network latency, wait, and full response (exp/analyze_results.py style) ---
    if len(df):
        df["net_lat"] = df["time_reception"] - df["time_emit"]
        df["wait"] = df["time_in"] - df["time_reception"]
        df["resp"] = df["time_out"] - df["time_emit"]
        AVGLAT = float(df["net_lat"].mean())
        AVGWAIT = float(df["wait"].mean())
        AVGRESP = float(df["resp"].mean())
    else:
        AVGLAT = AVGWAIT = AVGRESP = 0.0

    # --- Execution Time (YAFS: time_service = time_out - time_in) ---
    avg_execution_time = float(df["time_service"].mean())
    total_execution_time = float(df["time_service"].sum())

    # --- Resource Utilization (per-node: sum(time_service) / total_time) ---
    node_service_time = df.groupby("TOPO.dst")["time_service"].sum()
    node_utilization = node_service_time / total_time
    avg_utilization = float(node_utilization.mean()) if len(node_utilization) else 0.0
    max_utilization = float(node_utilization.max()) if len(node_utilization) else 0.0
    nodes_used = len(node_utilization)
    load_balance_std = float(node_service_time.std()) if len(node_service_time) > 1 else 0.0

    # --- Energy Consumption (exp/analyze_results.py: (service_time/1000)*power per node) ---
    total_energy = np.nan
    if cloud_id is not None and len(df):
        total_energy = 0.0
        for node_id, group in df.groupby("TOPO.dst"):
            service_time_ms = group["time_service"].sum()
            power = POWER_CLOUD if int(node_id) == int(cloud_id) else POWER_FOG
            total_energy += (service_time_ms / 1000.0) * power

    # --- Network (from link df) ---
    total_bytes = float(stats.df_link["size"].sum())
    avg_link_latency = float(stats.df_link["latency"].mean())
    avg_buffer = float(stats.df_link["buffer"].mean())

    # --- MISMATCH & module-level fog/cloud stats (exp/analyze_results.py style) ---
    MISMATCH = 0
    MISMATCH_PCT = 0.0
    FOG_PURE = CLOUD_INV = 0
    FOG_PCT = CLOUD_PCT = 0.0
    if planned_placement and cloud_id is not None and len(df):
        def _check_mismatch(row):
            mod = row.get("module")
            try:
                actual = int(row.get("TOPO.dst"))
            except (TypeError, ValueError):
                return 0
            planned = int(planned_placement.get(mod, cloud_id))
            return 1 if planned != cloud_id and actual == cloud_id else 0

        df["is_mismatch"] = df.apply(_check_mismatch, axis=1)
        MISMATCH = int(df["is_mismatch"].sum())
        MISMATCH_PCT = (MISMATCH / len(df)) * 100.0 if len(df) else 0.0

        mod_locs = df.groupby("module")["TOPO.dst"].unique()
        pure_fog = 0
        cloud_inv = 0
        for locs in mod_locs.values:
            if cloud_id in locs:
                cloud_inv += 1
            else:
                pure_fog += 1
        MOD_count = df["module"].nunique()
        FOG_PURE = pure_fog
        CLOUD_INV = cloud_inv
        FOG_PCT = (pure_fog / MOD_count) * 100.0 if MOD_count else 0.0
        CLOUD_PCT = (cloud_inv / MOD_count) * 100.0 if MOD_count else 0.0

    # --- SLA Violation (SLAV %) ---
    SLAV_PCT = 0.0
    if deadlines and len(df):
        def _check_slav(row):
            try:
                app_id = int(row.get("app"))
            except (TypeError, ValueError):
                return 0
            d = float(deadlines.get(app_id, 999999.0))
            return 1 if row["resp"] > d else 0

        SLAV_PCT = (df.apply(_check_slav, axis=1).sum() / len(df)) * 100.0

    # APP, MOD (exp-style: unique app and module count at runtime)
    APP = int(df["app"].nunique()) if "app" in df.columns and len(df) else 0
    MOD = int(df["module"].nunique()) if "module" in df.columns and len(df) else 0

    metrics = {
        "placement": placement_name,
        "total_requests": len(stats.df),
        "total_time": total_time,
        "APP": APP,
        "MOD": MOD,
        # Network / wait / response — satuan: ms (exp-style)
        "AVGLAT": AVGLAT,
        "AVGWAIT": AVGWAIT,
        "AVGRESP": AVGRESP,
        # Service Latency (YAFS, for internal/plot use)
        "avg_service_latency": avg_service_latency,
        "avg_total_response": avg_total_response,
        "max_service_latency": max_service_latency,
        # Execution Time
        "avg_execution_time": avg_execution_time,
        "total_execution_time": total_execution_time,
        # Resource Utilization
        "avg_utilization": avg_utilization,
        "max_utilization": max_utilization,
        "nodes_used": nodes_used,
        "load_balance_std": load_balance_std,
        # Energy Consumption
        "total_energy": total_energy,
        # Network
        "total_bytes_transmitted": total_bytes,
        "avg_link_latency": avg_link_latency,
        "avg_buffer": avg_buffer,
        "max_buffer": float(stats.df_link["buffer"].max()),
        # Mismatch / module-level fog-cloud stats
        "MISMATCH": MISMATCH,
        "MISMATCH_PCT": MISMATCH_PCT,
        "FOG_PURE": FOG_PURE,
        "FOG_PCT": FOG_PCT,
        "CLOUD_INV": CLOUD_INV,
        "CLOUD_PCT": CLOUD_PCT,
        # SLA violation
        "SLAV_PCT": SLAV_PCT,
    }
    return metrics


def compare_placements(
    placements=(
        "CNPlacement",
        "GAPlacement",
        "ILPPlacement",
        "GRPlacement",
        "RDMPlacement",
        "PSOPlacement",
        "CNGAPSOPlacement",
    ),
    duration=None,
    scenarios_dir=None,
):
    """Compare results from multiple placement algorithms using YAFS metrics."""
    print("=" * 70)
    print("FOG Computing Placement - Result Analysis (YAFS Metrics)")
    print("=" * 70)

    topology = load_topology(scenarios_dir)
    if topology is None:
        print("Warning: Topology not found; Energy Consumption will be NaN. Run generator/generate_scenario.py and ensure networkDefinition.json has WATT, model.")
    else:
        print("  ✓ Topology loaded (for energy metric)")

    scenarios_dir = _project_root / "scenarios" if scenarios_dir is None else scenarios_dir
    deadlines = load_app_deadlines(scenarios_dir)
    cloud_id = load_cloud_id(scenarios_dir)

    results = []
    for placement in placements:
        print(f"Loading {placement}...")
        df_trace, df_link, result_path = load_results(placement)
        if result_path is None:
            print(f"  ✗ {placement}: No results found")
            continue
        total_time = duration
        if total_time is None and df_trace is not None:
            total_time = float(df_trace["time_out"].max()) if len(df_trace) else None
        planned = load_planned_placement(placement, scenarios_dir, cloud_id=cloud_id)
        metrics = analyze_placement(
            placement,
            result_path,
            topology,
            total_time,
            deadlines=deadlines,
            planned_placement=planned,
            cloud_id=cloud_id,
        )
        if metrics is not None:
            results.append(metrics)
            print(f"  ✓ {placement}: {metrics['total_requests']} requests, total_time={metrics['total_time']:.0f}")
        else:
            print(f"  ✗ {placement}: Analysis failed")

    if not results:
        print("\nNo results to analyze. Run simulations first!")
        return

    df_results = pd.DataFrame(results)

    # Tabel utama (exp-style): hanya metrik dengan satuan seperti di exp/analyze_results.py
    print("\n" + "=" * 70)
    print("HASIL PERBANDINGAN (satuan: AVGLAT/AVGWAIT/AVGRESP ms, SLAV %, ENERGY Joule)")
    print("-" * 70)
    display = df_results[["placement", "APP", "MOD", "FOG_PURE", "CLOUD_INV", "MISMATCH", "AVGLAT", "AVGWAIT", "AVGRESP", "SLAV_PCT", "total_energy"]].copy()
    display.columns = ["placement", "APP", "MOD", "FOG (Pure)", "CLOUD (Inv.)", "MISMATCH", "AVGLAT (ms)", "AVGWAIT (ms)", "AVGRESP (ms)", "SLAV %", "ENERGY (Joule)"]
    print(display.to_string(index=False))

    print("\n📘 KETERANGAN PARAMETER:")
    print("APP            : Jumlah aplikasi unik yang dieksekusi dalam simulasi.")
    print("MOD            : Jumlah modul/service unik yang muncul pada runtime.")
    print("FOG (Pure)     : Modul yang seluruh eksekusinya hanya terjadi di Fog (tanpa Cloud).")
    print("CLOUD (Inv.)   : Modul yang pernah dieksekusi di Cloud minimal satu kali.")
    print("MISMATCH       : Request yang direncanakan di Fog namun dieksekusi di Cloud.")
    print("AVGLAT (ms)    : Rata-rata latensi jaringan (time_reception - time_emit).")
    print("AVGWAIT (ms)   : Rata-rata waktu tunggu dalam antrean komputasi.")
    print("AVGRESP (ms)   : Rata-rata waktu respon total (time_out - time_emit).")
    print("SLAV %         : Persentase request yang melanggar deadline SLA.")
    print("ENERGY (Joule) : Total energi yang dikonsumsi selama simulasi.")

    output_dir = _project_root / "analysis"
    output_dir.mkdir(exist_ok=True)
    out_csv = output_dir / "comparison_results.csv"
    df_results.to_csv(out_csv, index=False)
    print(f"\n✓ Results saved to: {out_csv}")

    try:
        generate_plots(df_results)
        print(f"✓ Plots saved to: {output_dir}/")
    except Exception as e:
        print(f"Warning: Could not generate plots: {e}")

    try:
        scenarios_dir = _project_root / "scenarios" if scenarios_dir is None else scenarios_dir
        generate_scenario_visualizations(scenarios_dir, df_results)
        print(f"✓ Scenario visualizations saved to: {output_dir}/")
    except Exception as e:
        print(f"Warning: Could not generate scenario visualizations: {e}")

    print("\n" + "=" * 70)


def plot_topology_layered(output_dir, topology_data):
    """Draw network topology in layered view: CLOUD, CFG, FOG (spring layout), FG."""
    if topology_data is None:
        return
    # Build NetworkX graph with node type
    G = nx.Graph()
    for e in topology_data["entity"]:
        nid = e["id"]
        t = e.get("type", "FOG")
        if t not in _LAYER_ORDER:
            t = "FOG"
        G.add_node(nid, type=t)
    for link in topology_data["link"]:
        G.add_edge(link["s"], link["d"])

    color_map = {"CLOUD": "green", "CFG": "orange", "FOG": "lightblue", "FG": "lightgreen"}
    cloud_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "CLOUD"]
    cfg_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "CFG"]
    fog_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "FOG"]
    fg_nodes = [n for n in G.nodes() if G.nodes[n].get("type") == "FG"]

    pos = {}
    # CLOUD
    y_cloud = 3.0
    for i, node in enumerate(cloud_nodes):
        pos[node] = (0, y_cloud)
    # CFG
    y_cfg = 2.3
    cfg_spacing = 2.5 / max(1, len(cfg_nodes) - 1) if len(cfg_nodes) > 1 else 0
    for i, node in enumerate(cfg_nodes):
        x_cfg = -1.25 + i * cfg_spacing if len(cfg_nodes) > 1 else 0
        pos[node] = (x_cfg, y_cfg)
    # FOG: spring layout on subgraph, then scale to layer area
    if fog_nodes:
        fog_subgraph = G.subgraph(fog_nodes)
        fog_pos = nx.spring_layout(fog_subgraph, k=0.8, iterations=100, seed=42)
        y_fog, fog_width, fog_height = 0.8, 4.0, 2.0
        for node in fog_nodes:
            x_scaled = fog_pos[node][0] * fog_width / 2
            y_scaled = fog_pos[node][1] * fog_height / 2 + y_fog
            pos[node] = (x_scaled, y_scaled)
    # FG
    y_fg = -0.6
    fg_spacing = 4.0 / max(1, len(fg_nodes) - 1) if len(fg_nodes) > 1 else 0
    for i, node in enumerate(fg_nodes):
        x_fg = -2.0 + i * fg_spacing if len(fg_nodes) > 1 else 0
        pos[node] = (x_fg, y_fg)

    node_colors = [color_map.get(G.nodes[n].get("type", "FOG"), "gray") for n in G.nodes()]
    node_sizes = [
        1200 if G.nodes[n].get("type") == "CLOUD" else 800 if G.nodes[n].get("type") in ("CFG", "FG") else 600
        for n in G.nodes()
    ]

    plt.figure(figsize=(18, 14))
    nx.draw(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=True,
        font_size=10,
        font_weight="bold",
        font_color="black",
        edge_color="lightgray",
        width=0.7,
        alpha=0.8,
    )
    plt.axhline(y=2.75, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    plt.axhline(y=2.0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    plt.axhline(y=-0.3, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    plt.text(-2.5, 3.0, "CLOUD LAYER", fontsize=12, fontweight="bold", ha="left")
    plt.text(-2.5, 2.3, "CFG LAYER", fontsize=12, fontweight="bold", ha="left")
    plt.text(-2.5, 0.8, "FOG LAYER", fontsize=12, fontweight="bold", ha="left")
    plt.text(-2.5, -0.6, "FG LAYER", fontsize=12, fontweight="bold", ha="left")
    legend_order = ["CLOUD", "CFG", "FOG", "FG"]
    legend_elements = [
        Patch(facecolor=color_map[t], label=f"{t} ({sum(1 for n in G.nodes() if G.nodes[n].get('type') == t)} nodes)")
        for t in legend_order
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize=12)
    plt.title("Network Topology (Layered View)", fontsize=18, fontweight="bold", pad=20)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_dir / "topology_layered.png", dpi=200, bbox_inches="tight")
    plt.close()


def plot_user_distribution_per_app(output_dir, users_data):
    """Bar chart: number of user sources per application."""
    if not users_data or "sources" not in users_data:
        return
    counts = defaultdict(int)
    for s in users_data["sources"]:
        app = s.get("app", "?")
        counts[str(app)] += 1
    apps = sorted(counts.keys(), key=lambda x: (int(x) if str(x).isdigit() else 999, x))
    vals = [counts[a] for a in apps]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(apps, vals, color="steelblue", edgecolor="gray")
    ax.set_xlabel("Application")
    ax.set_ylabel("Number of user sources")
    ax.set_title("User distribution per app")
    plt.tight_layout()
    plt.savefig(output_dir / "user_distribution_per_app.png", dpi=300)
    plt.close()


def plot_requests_per_app_from_trace(output_dir, placement_name):
    """Bar chart: number of processed requests per app from sim_trace (one placement)."""
    df_trace, _, _ = load_results(placement_name)
    if df_trace is None or "app" not in df_trace.columns:
        return
    counts = df_trace["app"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(counts.index.astype(str), counts.values, color="coral", edgecolor="gray")
    ax.set_xlabel("Application")
    ax.set_ylabel("Processed requests")
    ax.set_title(f"Requests per app (placement: {placement_name})")
    plt.tight_layout()
    plt.savefig(output_dir / "requests_per_app.png", dpi=300)
    plt.close()


def plot_latency_distribution(output_dir, placement_name):
    """Histogram of service latency (time_response) from one placement trace."""
    result_path = _project_root / "results" / placement_name / "sim_trace"
    if not result_path.with_suffix(".csv").exists():
        return
    try:
        stats = Stats(defaultPath=str(result_path))
        stats.compute_times_df()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(stats.df["time_response"].dropna(), bins=50, color="teal", edgecolor="white", alpha=0.8)
        ax.set_xlabel("Service latency (time_response)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Latency distribution ({placement_name})")
        plt.tight_layout()
        plt.savefig(output_dir / "latency_distribution.png", dpi=300)
        plt.close()
    except Exception:
        pass


def plot_metric_radar(output_dir, df_results):
    """Radar chart comparing placements on normalized metrics (latency, energy, utilization, etc.)."""
    if df_results is None or len(df_results) < 2:
        return
    metrics = ["avg_service_latency", "avg_execution_time", "total_energy", "load_balance_std"]
    labels = ["Service latency", "Execution time", "Energy", "Load imbalance"]
    df = df_results.copy()
    for c in metrics:
        if c not in df.columns or df[c].isna().all():
            return
    df = df.fillna(df[metrics].max())
    # Normalize to 0-1 (lower is better for all)
    for c in metrics:
        mx = df[c].max()
        mn = df[c].min()
        if mx > mn:
            df[c + "_n"] = 1 - (df[c] - mn) / (mx - mn)
        else:
            df[c + "_n"] = 1.0
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    for idx in df.index:
        row = df.loc[idx]
        name = row["placement"]
        vals = [row[m + "_n"] for m in metrics]
        vals += vals[:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=name)
        ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Placement comparison (normalized, higher is better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    plt.savefig(output_dir / "metric_radar.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_scenario_visualizations(scenarios_dir, df_results=None):
    """Generate topology, user distribution, and optional trace-based plots."""
    output_dir = _project_root / "analysis"
    output_dir.mkdir(exist_ok=True)
    topology_data = load_topology_data(scenarios_dir)
    users_data = load_users_data(scenarios_dir)
    plot_topology_layered(output_dir, topology_data)
    plot_user_distribution_per_app(output_dir, users_data)
    if df_results is not None and len(df_results) > 0:
        first_placement = str(df_results.iloc[0]["placement"])
        plot_requests_per_app_from_trace(output_dir, first_placement)
        plot_latency_distribution(output_dir, first_placement)
        try:
            plot_metric_radar(output_dir, df_results)
        except Exception:
            pass


def generate_plots(df_results):
    """Generate comparison plots for the four metrics."""
    output_dir = _project_root / "analysis"

    # 1. Service Latency
    plt.figure(figsize=(10, 6))
    plt.bar(df_results["placement"], df_results["avg_service_latency"])
    plt.xlabel("Placement Algorithm")
    plt.ylabel("Avg Service Latency")
    plt.title("Service Latency Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "latency_comparison.png", dpi=300)
    plt.close()

    # 2. Execution Time
    plt.figure(figsize=(10, 6))
    plt.bar(df_results["placement"], df_results["avg_execution_time"])
    plt.xlabel("Placement Algorithm")
    plt.ylabel("Avg Execution Time")
    plt.title("Execution Time Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "execution_time_comparison.png", dpi=300)
    plt.close()

    # 3. Resource Utilization (avg utilization per node)
    plt.figure(figsize=(10, 6))
    plt.bar(df_results["placement"], df_results["avg_utilization"])
    plt.xlabel("Placement Algorithm")
    plt.ylabel("Avg Node Utilization")
    plt.title("Resource Utilization Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "load_balance_comparison.png", dpi=300)
    plt.close()

    # 4. Energy Consumption
    energy = df_results["total_energy"].copy()
    energy = energy.fillna(0)  # for plotting
    plt.figure(figsize=(10, 6))
    plt.bar(df_results["placement"], energy)
    plt.xlabel("Placement Algorithm")
    plt.ylabel("Total Energy (Joule)")
    plt.title("Energy Consumption Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "energy_comparison.png", dpi=300)
    plt.close()

    # 5. Nodes used
    plt.figure(figsize=(10, 6))
    plt.bar(df_results["placement"], df_results["nodes_used"])
    plt.xlabel("Placement Algorithm")
    plt.ylabel("Number of Nodes Used")
    plt.title("Node Utilization Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "nodes_utilization.png", dpi=300)
    plt.close()


def main():
    placements = [
        "CNPlacement",
        "GAPlacement",
        "ILPPlacement",
        "GRPlacement",
        "RDMPlacement",
        "PSOPlacement",
        "CNGAPSOPlacement",
    ]
    duration = None
    args = sys.argv[1:]
    if args and args[0].isdigit():
        duration = float(args[0])
        args = args[1:]
    if args:
        placements = args
    compare_placements(placements=placements, duration=duration)


if __name__ == "__main__":
    main()
