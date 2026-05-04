"""
Multi-instance experiment runner — Pakpahan et al. (2025) [1].
14 problem instances (5–70 apps, step 5) x 10 runs each.

Flow:
  - Topology is generated ONCE (fixed seed) and shared across ALL runs,
    matching the paper's single network infrastructure.
  - Applications and users are re-generated per (num_apps, run) for variability.

Results saved to: results/apps_{N}/run_{R}/{algorithm}/.
Also writes results/apps_{N}/run_{R}/appDefinition.json (snapshot for analysis/SLAV).
"""
import json
import sys
import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from generator.generate_scenario import generate_topology, generate_applications, generate_users
from generator.generate_placements import generate_placement
from placements.heuristic.rdm_placement import RDMPlacement
from placements.heuristic.sm_placement import SMPlacement      # SortMatch
from placements.heuristic.ffha_placement import FFHAPlacement  # FirstFitHopAware
from placements.heuristic.hop2_placement import Hop2Placement
from placements.heuristic.hop3_placement import Hop3Placement
from placements.heuristic.fff_placement import FFFPlacement    # FrameworkFirstFit
from placements.metaheuristic.ga_placement import GAPlacement
from placements.combinatorial.aco_placement import ACOPlacement
from placements.combinatorial.sa_placement import SAPlacement
from placements.combinatorial.ts_placement import TSPlacement
from placements.metaheuristic.pso_placement import PSOPlacement
from placements.metaheuristic.gwo_placement import GWOPlacement
from placements.metaheuristic.woa_placement import WOAPlacement
from placements.multiobjective.nsga2_placement import NSGAIIPlacement
from placements.multiobjective.moead_placement import MOEADPlacement
from runner.run_simulation import run_simulation

# ---------------------------------------------------------------------------
# Experiment Parameters — Pakpahan et al. (2025) [1]
# ---------------------------------------------------------------------------
# NUM_APPS_RANGE = list(range(1100, 1501, 100))  # [5, 10, 15, ..., 70] — 14 instances
NUM_APPS_RANGE = [100, 500, 1000, 1500]  # [5, 10, 15, ..., 70] — 14 instances
RUNS_PER_INSTANCE = 10
SIM_DURATION = 10000

TOPOLOGY_SEED = 8

# Algorithms
PLACEMENTS = [
    # ("RDM",  RDMPlacement),
    ("SM",   SMPlacement),
    ("FFHA", FFHAPlacement),
    # ("Hop2", Hop2Placement),
    # ("Hop3", Hop3Placement),
    # ("FFF",  FFFPlacement),
    ("GA",   GAPlacement),
    ("ACO",   ACOPlacement),
    ("SA",    SAPlacement),
    ("TS",    TSPlacement),
    ("PSO",   PSOPlacement),
    ("GWO",   GWOPlacement),
    ("WOA",   WOAPlacement),
    ("NSGA2", NSGAIIPlacement),
    ("MOEAD", MOEADPlacement),
]


def _instance_scenarios_dir(base_scenarios_dir: Path, num_apps: int, run: int) -> Path:
    return base_scenarios_dir / f"apps_{num_apps}" / f"run_{run}"


def run_instance(num_apps, run, scenarios_dir, topology):
    """Generate apps/users, placements, and run all simulations for one instance."""
    seed = num_apps * 100 + run  # varies per (num_apps, run), NOT topology

    scenarios_dir = Path(scenarios_dir)
    scenarios_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save shared topology (already generated once in main)
    with open(scenarios_dir / "networkDefinition.json", "w") as f:
        json.dump(topology, f, indent=2)

    applications = generate_applications(seed=seed, num_apps=num_apps)
    with open(scenarios_dir / "appDefinition.json", "w") as f:
        json.dump(applications, f, indent=2)

    run_root = _project_root / "results" / f"apps_{num_apps}" / f"run_{run}"
    run_root.mkdir(parents=True, exist_ok=True)
    with open(run_root / "appDefinition.json", "w") as f:
        json.dump(applications, f, indent=2)

    users = generate_users(topology, applications, seed=seed)
    with open(scenarios_dir / "usersDefinition.json", "w") as f:
        json.dump(users, f, indent=2)

    # 2) Generate placements
    for name, placement_class in PLACEMENTS:
        alloc_file = scenarios_dir / f"allocDefinition{name}.json"
        generate_placement(placement_class, topology, applications, users, alloc_file)

    # 3) Run simulations
    failed = []
    for name, _ in PLACEMENTS:
        results_dir = _project_root / "results" / f"apps_{num_apps}" / f"run_{run}" / name
        try:
            run_simulation(
                f"{name}Placement",
                SIM_DURATION,
                results_dir=results_dir,
                scenarios_dir=scenarios_dir,
            )
        except Exception as e:
            print(f"  ERROR {name}Placement: {e}")
            failed.append(name)

    return failed


def _worker(task):
    """
    Worker entrypoint for multiprocessing (must be top-level for Windows).
    task: (num_apps, run, base_scenarios_dir, topology)
    """
    num_apps, run, base_scenarios_dir, topology = task
    base_scenarios_dir = Path(base_scenarios_dir)
    scenarios_dir = _instance_scenarios_dir(base_scenarios_dir, num_apps, run)
    failed = run_instance(num_apps, run, scenarios_dir, topology)
    return num_apps, run, failed


def main():
    parser = argparse.ArgumentParser(description="Multi-instance experiment runner")
    parser.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Number of parallel workers (default: CPU-1)",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Disable parallelism (debug/fallback)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=RUNS_PER_INSTANCE,
        help="Runs per instance (default from script constant)",
    )
    parser.add_argument(
        "--apps-min",
        type=int,
        default=min(NUM_APPS_RANGE),
        help="Minimum number of apps (inclusive)",
    )
    parser.add_argument(
        "--apps-max",
        type=int,
        default=max(NUM_APPS_RANGE),
        help="Maximum number of apps (inclusive)",
    )
    parser.add_argument(
        "--apps-step",
        type=int,
        default=(NUM_APPS_RANGE[1] - NUM_APPS_RANGE[0]) if len(NUM_APPS_RANGE) > 1 else 100,
        help="Step for number of apps",
    )
    args = parser.parse_args()

    scenarios_dir = _project_root / "scenarios"
    scenarios_dir.mkdir(exist_ok=True)

    # If the CLI range args are untouched, preserve the explicit app counts
    # from NUM_APPS_RANGE instead of inferring an arithmetic progression.
    default_apps_min = min(NUM_APPS_RANGE)
    default_apps_max = max(NUM_APPS_RANGE)
    default_apps_step = (NUM_APPS_RANGE[1] - NUM_APPS_RANGE[0]) if len(NUM_APPS_RANGE) > 1 else 100
    using_default_range_args = (
        args.apps_min == default_apps_min
        and args.apps_max == default_apps_max
        and args.apps_step == default_apps_step
    )
    num_apps_range = (
        list(NUM_APPS_RANGE)
        if using_default_range_args
        else list(range(args.apps_min, args.apps_max + 1, args.apps_step))
    )
    runs_per_instance = args.runs

    all_failed = []

    print("=" * 60)
    print("Multi-Instance Experiment Runner")
    print(f"  Instances : {len(num_apps_range)} ({num_apps_range[0]}–{num_apps_range[-1]} apps, step {args.apps_step})")
    print(f"  Runs each : {runs_per_instance}")
    print(f"  Duration  : {SIM_DURATION} time units")
    print(f"  Topo seed : {TOPOLOGY_SEED} (fixed for all runs)")
    if args.sequential:
        print("  Mode      : sequential")
    else:
        print(f"  Mode      : parallel (jobs={args.jobs})")
    print("=" * 60)

    # Generate topology ONCE — shared across all 140 simulations
    print(f"\nGenerating shared topology (seed={TOPOLOGY_SEED})...")
    topology = generate_topology(seed=TOPOLOGY_SEED)

    tasks = []
    for num_apps in num_apps_range:
        for run in range(1, runs_per_instance + 1):
            tasks.append((num_apps, run, str(scenarios_dir), topology))

    total = len(tasks)
    if args.sequential or args.jobs <= 1:
        done = 0
        for num_apps, run, base_scenarios_dir, topology in tasks:
            done += 1
            instance_dir = _instance_scenarios_dir(Path(base_scenarios_dir), num_apps, run)
            print(f"\n[{done}/{total}] apps={num_apps}, run={run}")
            failed = run_instance(num_apps, run, instance_dir, topology)
            if failed:
                all_failed.append((num_apps, run, failed))
    else:
        done = 0
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            future_map = {ex.submit(_worker, t): t[:2] for t in tasks}
            for fut in as_completed(future_map):
                num_apps, run = future_map[fut]
                done += 1
                try:
                    num_apps, run, failed = fut.result()
                    print(f"\n[{done}/{total}] DONE apps={num_apps}, run={run}")
                    if failed:
                        all_failed.append((num_apps, run, failed))
                except Exception as e:
                    print(f"\n[{done}/{total}] ERROR apps={num_apps}, run={run}: {e}")
                    all_failed.append((num_apps, run, ["__WORKER_FAILED__"]))

    print("\n" + "=" * 60)
    print("Experiment Complete!")
    if all_failed:
        print(f"  {len(all_failed)} failed instance(s):")
        for num_apps, run, failed in all_failed:
            print(f"    apps={num_apps}, run={run}: {failed}")
    else:
        print("  All instances completed successfully.")
    print("=" * 60)
    print(f"\nResults saved to: {_project_root / 'results'}/")


if __name__ == "__main__":
    main()
