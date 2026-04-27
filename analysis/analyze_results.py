#!/usr/bin/env python3
"""
Aggregate YAFS traces under results/ (sim_trace.csv, optional sim_trace_link.csv).

Layout:
  results/apps_<N>/run_<R>/<algo>/sim_trace.csv  — semua (N, run) dirata-rata per algoritma
  results/<AlgoPlacement>/sim_trace.csv          — satu skenario (pakai scenarios/appDefinition.json)

Kolom tabel: apps, algo, mod, fog_cloud, lat, wait, resp, hops, jain, slav, failed, scr.
failed = rata-rata (antar run) jumlah aplikasi yang tidak selesai ke modul sink (FAILED di analyze_trace), bukan %.
scr = rata-rata (antar run) success completion rate (% aplikasi selesai).
Per baris = (N, algoritma); metrik = rata-rata antar run untuk N itu.
lat, wait, resp = definisi di plot_mean_response_time_sweep.
Multi-instance: appDefinition di results/.../run_R/ bila ada, else replay seed N*100+run.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from analysis.plot_mean_response_time_sweep import (
    compute_response_components,
    mean_service_interval_ms,
)

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

_RE_APPS = re.compile(r"^apps_(\d+)$")
_RE_RUN = re.compile(r"^run_(\d+)$")


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_network(scenarios_dir: Path) -> Tuple[Optional[int], set[int]]:
    data = _load_json(scenarios_dir / "networkDefinition.json")
    if not data:
        return None, set()
    cloud_id: Optional[int] = None
    fog: set[int] = set()
    for e in data.get("entity", []):
        eid = int(e["id"])
        et = str(e.get("type", "")).upper()
        if et == "CLOUD" or e.get("model") == "cloud":
            cloud_id = eid
        else:
            fog.add(eid)
    if cloud_id is None and data.get("entity"):
        cloud_id = max(int(x["id"]) for x in data["entity"])
    return cloud_id, fog


def _sink_module_name(app: dict) -> str:
    msgs = app.get("message", [])
    nxt: Dict[str, str] = {}
    first: Optional[str] = None
    for m in msgs:
        s, d = m.get("s"), m.get("d")
        if s == "None" or s is None:
            first = str(d)
        else:
            nxt[str(s)] = str(d)
    if first is None:
        mods = app.get("module", [])
        return str(mods[-1]["name"]) if mods else ""
    cur = first
    chain = [cur]
    while cur in nxt:
        cur = nxt[cur]
        chain.append(cur)
    return chain[-1]


def metadata_from_app_list(data: List[Dict[str, Any]]) -> Tuple[Dict[int, float], Dict[int, str], int]:
    deadlines: Dict[int, float] = {}
    sinks: Dict[int, str] = {}
    for app in data:
        aid = int(app["id"])
        deadlines[aid] = float(app.get("deadline", 1e18))
        sinks[aid] = _sink_module_name(app)
    return deadlines, sinks, len(data)


def load_apps(scenarios_dir: Path) -> Tuple[Dict[int, float], Dict[int, str], int]:
    data = _load_json(scenarios_dir / "appDefinition.json")
    if not data:
        return {}, {}, 0
    return metadata_from_app_list(data)


@lru_cache(maxsize=512)
def _metadata_sweep_replay(n_apps: int, run: int) -> Tuple[Dict[int, float], Dict[int, str], int]:
    """Fallback bila appDefinition.json belum disimpan di folder run (eksperimen lama)."""
    with contextlib.redirect_stdout(io.StringIO()):
        from generator.generate_scenario import generate_applications

        data = generate_applications(seed=n_apps * 100 + run, num_apps=n_apps)
    return metadata_from_app_list(data)


def metadata_for_sweep_trace(
    trace_path: Path, n_apps: int, run: int
) -> Tuple[Dict[int, float], Dict[int, str], int]:
    """Deadline/sink: utamakan appDefinition di results/.../run_R/ (ditulis run_experiment)."""
    data = _load_json(trace_path.parent.parent / "appDefinition.json")
    if data:
        return metadata_from_app_list(data)
    return _metadata_sweep_replay(n_apps, run)


def normalize_algo(folder_name: str) -> str:
    return folder_name[:-9] if folder_name.endswith("Placement") else folder_name


def discover_traces(results_root: Path) -> List[Tuple[str, Path]]:
    if not results_root.is_dir():
        return []
    return [(normalize_algo(p.parent.name), p) for p in results_root.rglob("sim_trace.csv")]


def discover_sweep_traces(
    results_root: Path, algorithms: Optional[set[str]] = None
) -> List[Tuple[str, Path, int, int]]:
    """(algo, sim_trace path, n_apps, run) — hanya bawah results/apps_*/run_*/*/."""
    out: List[Tuple[str, Path, int, int]] = []
    if not results_root.is_dir():
        return out
    for apps_dir in sorted(results_root.iterdir()):
        if not apps_dir.is_dir():
            continue
        ma = _RE_APPS.match(apps_dir.name)
        if not ma:
            continue
        n_apps = int(ma.group(1))
        for run_dir in sorted(apps_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            mr = _RE_RUN.match(run_dir.name)
            if not mr:
                continue
            run_id = int(mr.group(1))
            for algo_dir in run_dir.iterdir():
                if not algo_dir.is_dir():
                    continue
                trace = algo_dir / "sim_trace.csv"
                if trace.is_file():
                    algo = normalize_algo(algo_dir.name)
                    if algorithms is not None and algo not in algorithms:
                        continue
                    out.append((algo, trace, n_apps, run_id))
    return out


def jain_fairness(counts: List[float]) -> float:
    if not counts:
        return 1.0
    x = np.asarray(counts, dtype=float)
    s2 = float(np.sum(x * x))
    if s2 <= 0:
        return 1.0
    return float((np.sum(x) ** 2) / (len(x) * s2))


def _mean_e2e_per_request(df: pd.DataFrame) -> float:
    if "id" not in df.columns or "time_emit" not in df.columns or "time_out" not in df.columns:
        return float("nan")
    g = df.groupby("id", as_index=False).agg(t0=("time_emit", "min"), t1=("time_out", "max"))
    d = (g["t1"] - g["t0"]).dropna()
    return float(d.mean()) if len(d) else float("nan")


def _avg_hops_from_link(link_csv: Path) -> float:
    if not link_csv.is_file():
        return float("nan")
    try:
        dfl = pd.read_csv(link_csv, low_memory=False)
    except Exception:
        return float("nan")
    if "id" not in dfl.columns or dfl.empty:
        return float("nan")
    per = dfl.groupby("id").size()
    return float(per.mean()) if len(per) else float("nan")


def analyze_trace(
    trace_csv: Path,
    cloud_id: Optional[int],
    fog_ids: set[int],
    deadlines: Dict[int, float],
    sinks: Dict[int, str],
    n_expected_apps: int,
) -> Optional[Dict[str, Any]]:
    try:
        df = pd.read_csv(trace_csv, low_memory=False)
    except Exception:
        return None
    if df.empty:
        return None

    need = ("app", "module", "time_emit", "time_reception", "time_in", "time_out", "service", "TOPO.dst")
    if any(c not in df.columns for c in need):
        return None

    df = df.copy()
    df["app_str"] = df["app"].astype(str)

    apps_seen = df["app_str"].nunique()
    mods_seen = df["module"].astype(str).nunique()

    completed = 0
    for aid, smod in sinks.items():
        sub = df[(df["app_str"] == str(aid)) & (df["module"].astype(str) == str(smod))]
        if sub.empty:
            continue
        if "type" in df.columns:
            if (sub["type"].astype(str) == "COMP_M").any():
                completed += 1
        else:
            completed += 1

    scr = (completed / n_expected_apps * 100.0) if n_expected_apps else 0.0
    failed = max(0, n_expected_apps - completed)

    fog_apps = cloud_apps = 0
    if cloud_id is not None:
        for a in df["app_str"].unique():
            nodes = df.loc[df["app_str"] == a, "TOPO.dst"].unique()
            if cloud_id in nodes:
                cloud_apps += 1
            else:
                fog_apps += 1
    tot_apps = fog_apps + cloud_apps
    fog_pct = (fog_apps / tot_apps * 100.0) if tot_apps else 0.0
    cloud_pct = (cloud_apps / tot_apps * 100.0) if tot_apps else 0.0

    link_csv = trace_csv.parent / "sim_trace_link.csv"
    avg_hops = _avg_hops_from_link(link_csv)

    jain = 0.0
    if fog_ids and "TOPO.dst" in df.columns:
        execs = df[df["TOPO.dst"].isin(fog_ids)].groupby("TOPO.dst").size()
        loads = [float(execs.get(n, 0)) for n in sorted(fog_ids)]
        jain = jain_fairness(loads)

    slav = 0.0
    if deadlines and "id" in df.columns:
        g = df.groupby(["app_str", "id"], as_index=False).agg(
            te=("time_emit", "min"), to=("time_out", "max")
        )
        g["resp"] = g["to"] - g["te"]
        violated = 0
        total = 0
        for a_str, sub in g.groupby("app_str"):
            try:
                aid = int(a_str)
            except ValueError:
                continue
            if aid not in deadlines:
                continue
            total += 1
            if float(sub["resp"].max()) > deadlines[aid]:
                violated += 1
        slav = (violated / total * 100.0) if total else 0.0

    return {
        "APP": int(apps_seen),
        "MOD": int(mods_seen),
        "SCR": float(scr),
        "FAILED": int(failed),
        "FOG_PCT": fog_pct,
        "CLOUD_PCT": cloud_pct,
        "AVG_HOPS": avg_hops,
        "JAINS_PLACEMENT": jain,
        "SLAV_PCT": slav,
    }


def per_trace_table_metrics(
    path: Path,
    cloud_id: Optional[int],
    fog_ids: set[int],
    deadlines: Dict[int, float],
    sinks: Dict[int, str],
    n_exp: int,
) -> Optional[Dict[str, float]]:
    st = analyze_trace(path, cloud_id, fog_ids, deadlines, sinks, n_exp)
    if st is None:
        return None
    comp = compute_response_components(path)
    if comp is None:
        return None
    resp = mean_service_interval_ms(path)
    if resp is None:
        return None
    return {
        "mod": float(st["MOD"]),
        "fog_pct": float(st["FOG_PCT"]),
        "cloud_pct": float(st["CLOUD_PCT"]),
        "lat": float(comp["latency"]),
        "wait": float(comp["wait"]),
        "resp": float(resp),
        "hops": float(st["AVG_HOPS"]),
        "jain": float(st["JAINS_PLACEMENT"]),
        "slav": float(st["SLAV_PCT"]),
        "failed": float(st["FAILED"]),
        "scr": float(st["SCR"]),
    }


def aggregate_by_apps_and_algo(
    raw: List[Tuple[int, str, Dict[str, float]]],
) -> List[Dict[str, Any]]:
    """Satu baris per (N aplikasi, algoritma); rata-rata hanya antar run untuk N tersebut."""
    acc: Dict[Tuple[int, str], List[Dict[str, float]]] = defaultdict(list)
    for n_apps, algo, row in raw:
        acc[(n_apps, algo)].append(row)
    out: List[Dict[str, Any]] = []
    for (n_apps, algo) in sorted(acc.keys(), key=lambda k: (k[0], k[1])):
        lst = acc[(n_apps, algo)]
        fp = float(np.mean([x["fog_pct"] for x in lst]))
        cp = float(np.mean([x["cloud_pct"] for x in lst]))
        out.append(
            {
                "apps": int(n_apps),
                "algo": algo,
                "mod": float(np.mean([x["mod"] for x in lst])),
                "fog_cloud": f"{fp:.1f}/{cp:.1f}",
                "lat": float(np.mean([x["lat"] for x in lst])),
                "wait": float(np.mean([x["wait"] for x in lst])),
                "resp": float(np.mean([x["resp"] for x in lst])),
                "hops": float(np.nanmean([x["hops"] for x in lst])),
                "jain": float(np.mean([x["jain"] for x in lst])),
                "slav": float(np.mean([x["slav"] for x in lst])),
                "failed": float(np.mean([x["failed"] for x in lst])),
                "scr": float(np.mean([x["scr"] for x in lst])),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Rangkum sim_trace: per (N aplikasi, algoritma) rata-rata antar run saja; "
            "layout apps_<N>/run_<R>/<algo> atau flat. lat/wait/resp = plot_mean_response_time_sweep."
        )
    )
    p.add_argument("--results-dir", type=Path, default=_ROOT / "results")
    p.add_argument("--scenarios-dir", type=Path, default=_ROOT / "scenarios")
    p.add_argument(
        "--algorithms",
        type=str,
        default=",".join(DEFAULT_ALGO_ORDER),
        help="Comma-separated algorithm folder names (default: DEFAULT_ALGO_ORDER in this file)",
    )
    p.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "analysis" / "summary_by_apps_and_algorithm.csv",
        help="Path CSV (kolom apps + algo + …)",
    )
    args = p.parse_args()

    algorithms = [a.strip() for a in str(args.algorithms).split(",") if a.strip()]
    algo_set = set(algorithms)

    cloud_id, fog_ids = load_network(args.scenarios_dir)

    sweep = discover_sweep_traces(args.results_dir, algorithms=algo_set)
    raw: List[Tuple[int, str, Dict[str, float]]] = []

    if sweep:
        for algo, path, n_apps, run in sorted(sweep, key=lambda x: (x[2], x[3], x[0], str(x[1]))):
            dl, sk, nexp = metadata_for_sweep_trace(path, n_apps, run)
            m = per_trace_table_metrics(path, cloud_id, fog_ids, dl, sk, nexp)
            if m is not None:
                raw.append((n_apps, algo, m))
    else:
        pairs = discover_traces(args.results_dir)
        if not pairs:
            print(f"No sim_trace.csv under {args.results_dir}")
            sys.exit(1)
        dl, sk, nexp = load_apps(args.scenarios_dir)
        for algo, path in sorted(pairs, key=lambda x: (x[0], str(x[1]))):
            if algo not in algo_set:
                continue
            m = per_trace_table_metrics(path, cloud_id, fog_ids, dl, sk, nexp)
            if m is not None:
                raw.append((int(nexp), algo, m))

    if not raw:
        print("No valid traces (check columns in sim_trace.csv and paths).")
        sys.exit(1)

    table = aggregate_by_apps_and_algo(raw)

    print("Summary: per (N apps, algorithm), mean over runs only")
    print(f"  results: {args.results_dir}")
    print(f"  traces : {len(raw)} (rows after aggregate: {len(table)})")
    print(f"  layout: {'apps_*/run_*' if sweep else 'flat'}")

    hdr = (
        f"{'apps':>5} {'algo':<8} {'mod':>6} {'fog/cld':>12} "
        f"{'lat':>8} {'wait':>7} {'resp':>7} "
        f"{'hops':>7} {'jain':>6} {'slav%':>6} {'fail':>6} {'scr%':>6}"
    )
    cur_apps: Optional[int] = None
    for r in sorted(table, key=lambda x: (x["apps"], x["algo"])):
        if r["apps"] != cur_apps:
            cur_apps = r["apps"]
            print()
            print(f"--- {cur_apps} applications (mean over runs) ---")
            print(hdr)
            print("-" * len(hdr))
        h = r["hops"]
        hs = f"{h:7.2f}" if np.isfinite(h) else "   nan"
        print(
            f"{r['apps']:5d} {r['algo']:<8} {r['mod']:6.1f} {r['fog_cloud']:>12} "
            f"{r['lat']:8.2f} {r['wait']:7.2f} {r['resp']:7.2f} "
            f"{hs} {r['jain']:6.3f} {r['slav']:5.1f} {r['failed']:6.2f} {r['scr']:6.1f}"
        )
    print()

    col_order = [
        "apps",
        "algo",
        "mod",
        "fog_cloud",
        "lat",
        "wait",
        "resp",
        "hops",
        "jain",
        "slav",
        "failed",
        "scr",
    ]
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table)[col_order].to_csv(args.csv, index=False)
    print(f"Wrote {args.csv}")


if __name__ == "__main__":
    main()
