"""
Simulation Runner for YAFS 3.1
Loads scenario JSON files and runs simulation with selected placement.

This is RUN 2: Execute simulation with pre-generated configurations.
"""
import json
import sys
import argparse
from pathlib import Path
import logging.config

# Ensure project root on path for imports when run as script
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import networkx as nx

from yafs.core import Sim
from yafs.application import create_applications_from_json
from yafs.topology import Topology
from yafs.placement import JSONPlacement
from runner.json_population import JSONPopulation
from runner.path_routing import create_routing_strategy


def load_scenario(scenarios_dir):
    """Load all scenario files."""
    with open(scenarios_dir / "networkDefinition.json") as f:
        topology_data = json.load(f)
    
    with open(scenarios_dir / "appDefinition.json") as f:
        applications_data = json.load(f)
    
    with open(scenarios_dir / "usersDefinition.json") as f:
        users_data = json.load(f)
    
    return topology_data, applications_data, users_data


def load_placement(scenarios_dir, placement_name):
    """Load placement allocation file."""
    alloc_name = placement_name.replace("Placement", "")
    alloc_file = scenarios_dir / f"allocDefinition{alloc_name}.json"
    
    if not alloc_file.exists():
        raise FileNotFoundError(f"Placement file not found: {alloc_file}")
    
    with open(alloc_file) as f:
        placement_data = json.load(f)
    
    # YAFS uses app name as string (e.g. "0", "1"); ensure allocation matches
    for item in placement_data.get("initialAllocation", []):
        if "app" in item and not isinstance(item["app"], str):
            item["app"] = str(item["app"])
    
    return placement_data


def run_simulation(placement_name, stop_time=20000, routing: str = "device_speed", results_dir=None):
    """
    Run simulation with specified placement algorithm.

    Args:
        placement_name: Name of placement (e.g., "CNPlacement", "GAPlacement")
        stop_time: Simulation duration in time units
        routing: Path routing strategy
        results_dir: Override output directory (used by multi-instance runner)
    """
    print("=" * 60)
    print(f"Running YAFS Simulation: {placement_name}")
    print("=" * 60)

    # Paths
    project_root = Path(__file__).parent.parent
    scenarios_dir = project_root / "scenarios"
    if results_dir is None:
        results_dir = project_root / "results" / placement_name
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load scenario
    print("\nLoading scenario...")
    topology_data, applications_data, users_data = load_scenario(scenarios_dir)
    print(f"  ✓ Topology: {len(topology_data['entity'])} nodes, {len(topology_data['link'])} links")
    print(f"  ✓ Applications: {len(applications_data)} apps")
    print(f"  ✓ Users: {len(users_data['sources'])} sources")
    
    # Load placement
    print(f"\nLoading placement: {placement_name}...")
    placement_data = load_placement(scenarios_dir, placement_name)
    print(f"  ✓ Allocations: {len(placement_data['initialAllocation'])} modules")
    
    # Create topology
    print("\nCreating topology...")
    t = Topology()
    t.load(topology_data)
    print(f"  ✓ Topology loaded")
    
    # Create applications
    print("\nCreating applications...")
    apps = create_applications_from_json(applications_data)
    print(f"  ✓ {len(apps)} applications created")
    
    # Create placement
    print(f"\nSetting up placement: {placement_name}...")
    placement = JSONPlacement(name=placement_name, json=placement_data)
    print(f"  ✓ Placement configured")
    
    # Create routing
    print("\nSetting up routing...")
    selectorPath = create_routing_strategy(routing)
    print(f"  ✓ Routing configured: {routing}")
    
    # Create simulator
    print("\nInitializing simulator...")
    result_file = str(results_dir / "sim_trace")
    s = Sim(t, default_results_path=result_file)
    print(f"  ✓ Simulator initialized")
    
    # Population: sources from usersDefinition.json (exponential inter-arrival)
    pop = JSONPopulation(name="Statical", json_data=users_data, iteration=1)

    # Deploy applications + population (per app, via deploy_app2)
    print("\nDeploying applications and users...")
    for app_name in apps.keys():
        pop_app = JSONPopulation(name=f"Statical_{app_name}", json_data={}, iteration=1)
        pop_app.data = {"sources": [e for e in pop.data["sources"] if e["app"] == app_name]}
        s.deploy_app2(apps[app_name], placement, pop_app, selectorPath)
    print(f"  ✓ {len(apps)} applications deployed with {len(users_data['sources'])} sources (exponential)")
    
    # Run simulation
    print("\n" + "=" * 60)
    print(f"Starting simulation (duration: {stop_time} time units)...")
    print("=" * 60)
    
    s.run(stop_time)
    
    print("\n" + "=" * 60)
    print("Simulation Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - sim_trace.csv (module processing)")
    print(f"  - sim_trace_link.csv (network transmission)")
    
    return results_dir


def main():
    """Main simulation runner."""
    parser = argparse.ArgumentParser(description="Run YAFS fog computing simulation")
    parser.add_argument(
        "--placement",
        type=str,
        default="CNPlacement",
        choices=[
            "CNPlacement",
            "GAPlacement",
            "ILPPlacement",
            "GRPlacement",
            "RDMPlacement",
            "PSOPlacement",
            "CNGAPSOPlacement",
        ],
        help="Placement algorithm to use",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20000,
        help="Simulation duration in time units",
    )
    parser.add_argument(
        "--routing",
        type=str,
        default="device_speed",
        choices=["device_speed", "weighted_latency", "load_aware"],
        help="Path routing strategy to use",
    )
    
    args = parser.parse_args()

    try:
        run_simulation(args.placement, args.duration, args.routing)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
