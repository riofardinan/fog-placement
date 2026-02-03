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


def analyze_placement(placement_name, result_path, topology, total_time=None):
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

    # --- Service Latency (YAFS: time_response = time_out - time_reception; time_total_response) ---
    avg_service_latency = float(stats.df["time_response"].mean())
    avg_total_response = float(stats.df["time_total_response"].mean())
    max_service_latency = float(stats.df["time_response"].max())

    # --- Execution Time (YAFS: time_service = time_out - time_in) ---
    avg_execution_time = float(stats.df["time_service"].mean())
    total_execution_time = float(stats.df["time_service"].sum())

    # --- Resource Utilization (per-node: sum(time_service) / total_time) ---
    node_service_time = stats.df.groupby("TOPO.dst")["time_service"].sum()
    node_utilization = node_service_time / total_time
    avg_utilization = float(node_utilization.mean()) if len(node_utilization) else 0.0
    max_utilization = float(node_utilization.max()) if len(node_utilization) else 0.0
    nodes_used = len(node_utilization)
    load_balance_std = float(node_service_time.std()) if len(node_service_time) > 1 else 0.0

    # --- Energy Consumption (YAFS Stats.get_watt; requires topology with WATT, model) ---
    total_energy = np.nan
    if topology is not None:
        try:
            watt_results = stats.get_watt(total_time, topology, by=Metrics.WATT_SERVICE)
            total_energy = sum(v["watt"] for v in watt_results.values())
        except (KeyError, TypeError) as e:
            total_energy = np.nan  # topology missing WATT/model or old scenario

    # --- Network (from link df) ---
    total_bytes = float(stats.df_link["size"].sum())
    avg_link_latency = float(stats.df_link["latency"].mean())
    avg_buffer = float(stats.df_link["buffer"].mean())

    metrics = {
        "placement": placement_name,
        "total_requests": len(stats.df),
        "total_time": total_time,
        # Service Latency
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
    }
    return metrics


def compare_placements(
    placements=("CNPlacement", "GAPlacement", "ILPPlacement", "RLPlacement", "GNNPlacement"),
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
        metrics = analyze_placement(placement, result_path, topology, total_time)
        if metrics is not None:
            results.append(metrics)
            print(f"  ✓ {placement}: {metrics['total_requests']} requests, total_time={metrics['total_time']:.0f}")
        else:
            print(f"  ✗ {placement}: Analysis failed")

    if not results:
        print("\nNo results to analyze. Run simulations first!")
        return

    df_results = pd.DataFrame(results)

    print("\n" + "=" * 70)
    print("1. SERVICE LATENCY (lower is better)")
    print("-" * 70)
    cols = ["placement", "avg_service_latency", "avg_total_response", "max_service_latency"]
    print(df_results[cols].to_string(index=False))

    print("\n2. EXECUTION TIME (lower is better)")
    print("-" * 70)
    cols = ["placement", "avg_execution_time", "total_execution_time"]
    print(df_results[cols].to_string(index=False))

    print("\n3. RESOURCE UTILIZATION")
    print("-" * 70)
    cols = ["placement", "nodes_used", "avg_utilization", "max_utilization", "load_balance_std"]
    print(df_results[cols].to_string(index=False))

    print("\n4. ENERGY CONSUMPTION (lower is better)")
    print("-" * 70)
    cols = ["placement", "total_energy"]
    print(df_results[cols].to_string(index=False))

    print("\n5. NETWORK")
    print("-" * 70)
    cols = ["placement", "total_bytes_transmitted", "avg_link_latency", "avg_buffer"]
    print(df_results[cols].to_string(index=False))

    print("\n6. BEST PERFORMER PER METRIC")
    print("-" * 70)
    valid = df_results["avg_service_latency"].notna()
    if valid.any():
        best = df_results.loc[df_results.loc[valid, "avg_service_latency"].idxmin(), "placement"]
        print(f"  • Lowest Service Latency: {best}")
    valid = df_results["avg_execution_time"].notna()
    if valid.any():
        best = df_results.loc[df_results.loc[valid, "avg_execution_time"].idxmin(), "placement"]
        print(f"  • Lowest Execution Time: {best}")
    valid = df_results["load_balance_std"].notna()
    if valid.any():
        best = df_results.loc[df_results.loc[valid, "load_balance_std"].idxmin(), "placement"]
        print(f"  • Best Load Balance: {best}")
    if df_results["total_energy"].notna().any():
        best = df_results.loc[df_results["total_energy"].idxmin(), "placement"]
        print(f"  • Lowest Energy: {best}")

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
    plt.ylabel("Total Energy (WATT·time)")
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
    placements = ["CNPlacement", "GAPlacement", "ILPPlacement", "RLPlacement", "GNNPlacement"]
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
