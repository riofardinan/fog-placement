"""
Mean response time sweep plot (Pakpahan-style line chart).

Reads multi-instance experiment output::

    results/apps_{N}/run_{R}/{algorithm}/sim_trace.csv

For each trace, computes the end-to-end response time per request
``max(time_out) - min(time_emit)`` (= Service End Time − Emit Time), then:

1. **Run level:** mean over all rows in that ``sim_trace.csv``.
2. **Scenario level (N apps):** mean of those values across ``run_1`` … ``run_R``.
3. **Line chart:** x = application count ``N``, y = scenario-level mean; one line per algorithm.

Usage::

    python analysis/plot_mean_response_time_sweep.py
    python analysis/plot_mean_response_time_sweep.py --results-dir /path/to/results --output analysis/mean_rt_sweep.png

Requires: pandas, matplotlib, numpy (see requirements.txt).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Folder names under each run_* (must match runner/run_experiment.py PLACEMENTS short names)
DEFAULT_ALGO_ORDER = [
    "SM", 
    # "Hop3", 
    "FFHA", 
    # "FFF", 
    # "Hop2", 
    "GA", 
    # "RDM"
    "ACO",
    "SA",
    "TS",
    "PSO",
    "GWO",
    "WOA",
    "NSGA2",
    "MOEAD",
]

# Paper-style display names (figure legend)
DISPLAY_NAMES: Dict[str, str] = {
    # "RDM": "Random",
    "SM": "SortMatch",
    # "FFHA": "FirstFitHopAware",
    # "FFF": "FrameworkFirstFit",
    # "Hop2": "Hop2",
    # "Hop3": "Hop3",
    "GA": "GA",
    "ACO": "ACO",
    "SA": "SA",
    "TS": "TS",
    "PSO": "PSO",
    "GWO": "GWO",
    "WOA": "WOA",
    "NSGA2": "NSGA2",
    "MOEAD": "MOEAD",
}

# Approximate colours/markers to match comparative plots
STYLE: Dict[str, Tuple[str, str, str]] = {
    "SM": ("#1f77b4", "-", "o"),
    # "Hop3": ("#ff7f0e", "--", "s"),
    # "FFHA": ("#d62728", "-.", "^"),
    # "FFF": ("#8c564b", ":", "d"),
    # "Hop2": ("#e377c2", "-.", "v"),
    "GA": ("#bcbd22", "--", "p"),
    # "RDM": ("#17becf", "-.", "*"),
    "ACO": ("#17becf", "-.", "*"),
    "SA": ("#ff7f0e", "-.", "*"),
    "TS": ("#d62728", "-.", "*"),
    "PSO": ("#8c564b", "-.", "*"),
    "GWO": ("#e377c2", "-.", "*"),
    "WOA": ("#9467bd", "-.", "*"),
    "NSGA2": ("#ffbb78", "-.", "*"),
    "MOEAD": ("#7f7f7f", "-.", "*"),
}

_APPS_DIR = re.compile(r"^apps_(\d+)$")
_RUN_DIR = re.compile(r"^run_(\d+)$")


def mean_service_interval_ms(trace_csv: Path) -> Optional[float]:
    """
    Mean of End-to-End Response Time (max time_out - min time_emit) in milliseconds.
    """
    if not trace_csv.is_file():
        return None
    try:
        df = pd.read_csv(trace_csv, low_memory=False)
    except Exception:
        return None
        
    for col in ("time_out", "time_emit", "id"):
        if col not in df.columns:
            return None
            
    # Mengelompokkan berdasarkan 'id' request (atau topologi pesan YAFS)
    # Untuk mendapatkan waktu paling awal request dikirim dan paling akhir diselesaikan
    grouped = df.groupby('id').agg(
        start_time=('time_emit', 'min'),
        end_time=('time_out', 'max')
    )
    
    # Total Response Time = Service End Time - Emit Time
    delta = grouped['end_time'] - grouped['start_time']
    delta = delta.dropna()
    
    if len(delta) == 0:
        return None
        
    return float(delta.mean())


def discover_sweep(
    results_root: Path,
    algorithms: List[str],
) -> Tuple[List[int], Dict[str, Dict[int, List[float]]]]:
    """
    Scan results_root for apps_* / run_* / {algo}/sim_trace.csv.

    Returns:
        sorted list of app counts N
        run_means[algo][N] = list of per-run means (one float per successful run)
    """
    run_means: Dict[str, Dict[int, List[float]]] = {a: {} for a in algorithms}
    app_counts: set[int] = set()

    if not results_root.is_dir():
        return [], run_means

    for apps_dir in sorted(results_root.iterdir()):
        if not apps_dir.is_dir():
            continue
        m = _APPS_DIR.match(apps_dir.name)
        if not m:
            continue
        n_apps = int(m.group(1))
        app_counts.add(n_apps)

        for run_dir in sorted(apps_dir.iterdir()):
            if not run_dir.is_dir() or not _RUN_DIR.match(run_dir.name):
                continue
            for algo in algorithms:
                trace = run_dir / algo / "sim_trace.csv"
                v = mean_service_interval_ms(trace)
                if v is None:
                    continue
                run_means.setdefault(algo, {}).setdefault(n_apps, []).append(v)

    return sorted(app_counts), run_means


def aggregate_per_scenario(
    run_means: Dict[str, Dict[int, List[float]]],
    algorithms: List[str],
    app_counts: List[int],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    For each algorithm and N apps: y = mean(per-run means), err = std(per-run means, ddof=1).
    Missing (N, algo) -> nan.
    """
    y_series: Dict[str, np.ndarray] = {}
    err_series: Dict[str, np.ndarray] = {}
    for algo in algorithms:
        ys = []
        es = []
        for n in app_counts:
            runs = run_means.get(algo, {}).get(n, [])
            if not runs:
                ys.append(np.nan)
                es.append(np.nan)
            elif len(runs) == 1:
                ys.append(float(np.mean(runs)))
                es.append(0.0)
            else:
                ys.append(float(np.mean(runs)))
                es.append(float(np.std(runs, ddof=1)))
        y_series[algo] = np.array(ys, dtype=float)
        err_series[algo] = np.array(es, dtype=float)
    return y_series, err_series


def plot_sweep(
    app_counts: List[int],
    y_series: Dict[str, np.ndarray],
    err_series: Dict[str, np.ndarray],
    algorithms: List[str],
    output_path: Path,
    title: str = "Mean response time vs application count",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.array(app_counts, dtype=float)

    for algo in algorithms:
        y = y_series.get(algo)
        if y is None or np.all(np.isnan(y)):
            continue
        label_name = DISPLAY_NAMES.get(algo, algo)
        yvals = y
        # Legend: mean ± std over the 14 (or fewer) scenario points, like paper caption
        valid = yvals[~np.isnan(yvals)]
        if len(valid):
            mu = float(np.mean(valid))
            sig = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
            leg = f"{label_name} (avg={mu:.2f} ± {sig:.2f})"
        else:
            leg = label_name
        color, ls, marker = STYLE.get(algo, ("#333333", "-", "o"))
        ax.plot(
            x,
            yvals,
            color=color,
            linestyle=ls,
            marker=marker,
            markersize=6,
            label=leg,
            linewidth=1.8,
        )

    ax.set_xlabel("Application Count")
    ax.set_ylabel("Mean Response Time (ms)")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_summary_csv(
    app_counts: List[int],
    y_series: Dict[str, np.ndarray],
    err_series: Dict[str, np.ndarray],
    algorithms: List[str],
    csv_path: Path,
) -> None:
    rows = []
    for algo in algorithms:
        for i, n in enumerate(app_counts):
            rows.append(
                {
                    "algorithm": algo,
                    "display_name": DISPLAY_NAMES.get(algo, algo),
                    "num_apps": n,
                    "mean_response_time_ms": y_series[algo][i],
                    "std_across_runs_ms": err_series[algo][i],
                }
            )
    pd.DataFrame(rows).to_csv(csv_path, index=False)


def compute_response_components(trace_csv: Path) -> Optional[Dict[str, float]]:
    """
    Compute avg response time components per request from sim_trace.csv.

    Returns dict with keys: service, latency, wait (all in ms).

    Definitions (matching Pakpahan 2025 / YAFS semantics):
    - service = sum(time_out - time_in) per request       [CPU processing time]
    - latency = sum(time_reception - time_emit) per req   [network transit to node]
    - wait    = sum(time_in - time_reception) per req     [DES module queue wait]
    """
    if not trace_csv.is_file():
        return None
    try:
        df = pd.read_csv(trace_csv, low_memory=False)
    except Exception:
        return None
    required = ("time_out", "time_emit", "time_in", "time_reception", "service", "id")
    if any(c not in df.columns for c in required):
        return None

    df = df.copy()
    df["_lat"] = df["time_reception"] - df["time_emit"]
    df["_wt"]  = df["time_in"] - df["time_reception"]

    per_req = df.groupby("id").agg(
        svc=("service", "sum"),
        lat=("_lat",    "sum"),
        wt= ("_wt",     "sum"),
    ).dropna()

    if per_req.empty:
        return None
    return {
        "service": float(per_req["svc"].mean()),
        "latency": float(per_req["lat"].mean()),
        "wait":    float(per_req["wt"].mean()),
    }


def plot_response_components_bar(
    results_root: Path,
    algorithms: List[str],
    output_path: Path,
) -> None:
    """
    Fig. 9 style stacked bar chart: Avg Service Time / Latency / Wait Time per algorithm.
    Averaged across ALL runs and problem instances.
    X-axis order matches paper Fig. 9: FFHA, FFF, GA, Hop2, Hop3, RDM, SM.
    Component values shown inside each bar segment (like paper).
    """
    # X-axis order matching paper Fig. 9
    # fig9_order = ["FFHA", "FFF", "GA", "Hop2", "Hop3", "RDM", "SM"]
    plot_algos = [a for a in DEFAULT_ALGO_ORDER if a in algorithms]

    algo_components: Dict[str, Dict[str, List[float]]] = {
        a: {"service": [], "latency": [], "wait": []} for a in plot_algos
    }

    if not results_root.is_dir():
        return

    for apps_dir in sorted(results_root.iterdir()):
        if not apps_dir.is_dir() or not _APPS_DIR.match(apps_dir.name):
            continue
        for run_dir in sorted(apps_dir.iterdir()):
            if not run_dir.is_dir() or not _RUN_DIR.match(run_dir.name):
                continue
            for algo in plot_algos:
                trace = run_dir / algo / "sim_trace.csv"
                comp = compute_response_components(trace)
                if comp is None:
                    continue
                for k in ("service", "latency", "wait"):
                    algo_components[algo][k].append(comp[k])

    labels = [DISPLAY_NAMES.get(a, a) for a in plot_algos]
    svc_vals = [np.mean(algo_components[a]["service"]) if algo_components[a]["service"] else 0.0 for a in plot_algos]
    lat_vals = [np.mean(algo_components[a]["latency"]) if algo_components[a]["latency"] else 0.0 for a in plot_algos]
    wt_vals  = [np.mean(algo_components[a]["wait"])    if algo_components[a]["wait"]    else 0.0 for a in plot_algos]

    x = np.arange(len(plot_algos))
    fig, ax = plt.subplots(figsize=(11, 6))

    ax.bar(x, svc_vals, label="Avg. Service Time", color="#6baed6")
    ax.bar(x, lat_vals, bottom=svc_vals,           label="Avg. Latency",    color="#74c476")
    ax.bar(x, wt_vals,  bottom=[s + l for s, l in zip(svc_vals, lat_vals)],
           label="Avg. Wait Time", color="#fb6a4a")

    # Labels inside each bar segment (like paper Fig. 9) and total on top
    for xi, (svc, lat, wt) in enumerate(zip(svc_vals, lat_vals, wt_vals)):
        total = svc + lat + wt
        # Service label (bottom segment, centred)
        if svc > 15:
            ax.text(xi, svc / 2, f"{svc:.2f}", ha="center", va="center", fontsize=7.5, color="black")
        # Latency label (middle segment)
        if lat > 15:
            ax.text(xi, svc + lat / 2, f"{lat:.2f}", ha="center", va="center", fontsize=7.5, color="black")
        # Wait label (top segment)
        if wt > 15:
            ax.text(xi, svc + lat + wt / 2, f"{wt:.2f}", ha="center", va="center", fontsize=7.5, color="black")
        # Total above bar
        ax.text(xi, total + 8, f"{total:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Response Time (ms)")
    ax.set_title("Fig. 9. Response time components of algorithms.")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_project_root / "results",
        help="Root folder containing apps_<N>/run_<R>/<algo>/sim_trace.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_project_root / "analysis" / "mean_response_time_by_apps.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional path to write summary CSV (default: alongside PNG with .csv suffix)",
    )
    parser.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_ALGO_ORDER),
        help="Comma-separated algorithm folder names (e.g. SM,Hop3,FFHA,FFF,Hop2,GA,RDM)",
    )
    args = parser.parse_args()
    algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]

    app_counts, run_means = discover_sweep(args.results_dir, algorithms)
    if not app_counts:
        print(
            f"No data found under {args.results_dir} (expected apps_<N>/run_<R>/<algo>/sim_trace.csv).",
            file=sys.stderr,
        )
        return 1

    y_series, err_series = aggregate_per_scenario(run_means, algorithms, app_counts)
    csv_path = args.csv
    if csv_path is None:
        csv_path = args.output.with_suffix(".csv")
    write_summary_csv(app_counts, y_series, err_series, algorithms, csv_path)

    plot_sweep(
        app_counts,
        y_series,
        err_series,
        algorithms,
        args.output,
        title="Mean response time (mean of time_out − time_in) vs application count",
    )
    print(f"Wrote plot: {args.output}")
    print(f"Wrote table: {csv_path}")

    # Fig. 9 style stacked bar chart
    fig9_path = args.output.parent / "response_components_by_algo.png"
    plot_response_components_bar(args.results_dir, algorithms, fig9_path)
    print(f"Wrote Fig.9 style chart: {fig9_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())