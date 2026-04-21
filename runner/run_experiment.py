"""
Multi-instance experiment runner — Pakpahan et al. (2025) [1].
14 problem instances (5–70 apps, step 5) x 10 runs each.

Flow:
  - Topology is generated ONCE (fixed seed) and shared across ALL runs,
    matching the paper's single network infrastructure.
  - Applications and users are re-generated per (num_apps, run) for variability.

Results saved to: results/apps_{N}/run_{R}/{algorithm}/
"""
import json
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from generator.generate_scenario import generate_topology, generate_applications, generate_users
from generator.generate_placements import generate_placement
from placements.rdm_placement import RDMPlacement
from placements.sm_placement import SMPlacement      # SortMatch
from placements.ffha_placement import FFHAPlacement  # FirstFitHopAware
from placements.hop2_placement import Hop2Placement
from placements.hop3_placement import Hop3Placement
from placements.fff_placement import FFFPlacement    # FrameworkFirstFit
from placements.ga_placement import GAPlacement
from runner.run_simulation import run_simulation

# ---------------------------------------------------------------------------
# Experiment Parameters — Pakpahan et al. (2025) [1]
# ---------------------------------------------------------------------------
NUM_APPS_RANGE = list(range(5, 75, 5))   # [5, 10, 15, ..., 70] — 14 instances
RUNS_PER_INSTANCE = 10
SIM_DURATION = 10000

TOPOLOGY_SEED = 42

# 7 algorithms from the paper
PLACEMENTS = [
    ("RDM",  RDMPlacement),
    ("SM",   SMPlacement),
    ("FFHA", FFHAPlacement),
    ("Hop2", Hop2Placement),
    ("Hop3", Hop3Placement),
    ("FFF",  FFFPlacement),
    ("GA",   GAPlacement),
]


def run_instance(num_apps, run, scenarios_dir, topology):
    """Generate apps/users, placements, and run all simulations for one instance."""
    seed = num_apps * 100 + run  # varies per (num_apps, run), NOT topology

    # 1) Save shared topology (already generated once in main)
    with open(scenarios_dir / "networkDefinition.json", "w") as f:
        json.dump(topology, f, indent=2)

    applications = generate_applications(seed=seed, num_apps=num_apps)
    with open(scenarios_dir / "appDefinition.json", "w") as f:
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
            run_simulation(f"{name}Placement", SIM_DURATION, results_dir=results_dir)
        except Exception as e:
            print(f"  ERROR {name}Placement: {e}")
            failed.append(name)

    return failed


def main():
    scenarios_dir = _project_root / "scenarios"
    scenarios_dir.mkdir(exist_ok=True)

    total = len(NUM_APPS_RANGE) * RUNS_PER_INSTANCE
    done = 0
    all_failed = []

    print("=" * 60)
    print("Multi-Instance Experiment Runner")
    print(f"  Instances : {len(NUM_APPS_RANGE)} ({NUM_APPS_RANGE[0]}–{NUM_APPS_RANGE[-1]} apps, step 5)")
    print(f"  Runs each : {RUNS_PER_INSTANCE}")
    print(f"  Total     : {total}")
    print(f"  Duration  : {SIM_DURATION} time units")
    print(f"  Topo seed : {TOPOLOGY_SEED} (fixed for all runs)")
    print("=" * 60)

    # Generate topology ONCE — shared across all 140 simulations
    print(f"\nGenerating shared topology (seed={TOPOLOGY_SEED})...")
    topology = generate_topology(seed=TOPOLOGY_SEED)

    for num_apps in NUM_APPS_RANGE:
        for run in range(1, RUNS_PER_INSTANCE + 1):
            done += 1
            print(f"\n[{done}/{total}] apps={num_apps}, run={run}")
            failed = run_instance(num_apps, run, scenarios_dir, topology)
            if failed:
                all_failed.append((num_apps, run, failed))

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
