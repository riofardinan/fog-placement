"""
Placement Generator for YAFS 3.1
"""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Paper algorithms — Pakpahan et al. (2025)
from placements.heuristic.rdm_placement import RDMPlacement
from placements.heuristic.sm_placement import SMPlacement      # SortMatch
from placements.heuristic.ffha_placement import FFHAPlacement  # FirstFitHopAware
from placements.heuristic.hop2_placement import Hop2Placement
from placements.heuristic.hop3_placement import Hop3Placement
from placements.heuristic.fff_placement import FFFPlacement    # FrameworkFirstFit
from placements.metaheuristic.ga_placement import GAPlacement
# Other algorithms
from placements.experimental.gr_placement import GRPlacement      # Greedy
from placements.experimental.cn_placement import CNPlacement
from placements.combinatorial.ilp_placement import ILPPlacement
from placements.metaheuristic.pso_placement import PSOPlacement
from placements.metaheuristic.cngapso_placement import CNGAPSOPlacement
from placements.combinatorial.aco_placement import ACOPlacement
from placements.combinatorial.sa_placement import SAPlacement
from placements.combinatorial.ts_placement import TSPlacement
from placements.metaheuristic.gwo_placement import GWOPlacement
from placements.metaheuristic.woa_placement import WOAPlacement
from placements.multiobjective.nsga2_placement import NSGAIIPlacement
from placements.multiobjective.moead_placement import MOEADPlacement


def load_scenario():
    """Load generated scenario files."""
    scenarios_dir = Path(__file__).parent.parent / "scenarios"
    
    with open(scenarios_dir / "networkDefinition.json") as f:
        topology = json.load(f)
    
    with open(scenarios_dir / "appDefinition.json") as f:
        applications = json.load(f)
    
    with open(scenarios_dir / "usersDefinition.json") as f:
        users = json.load(f)
    
    return topology, applications, users


def generate_placement(placement_class, topology, applications, users, output_file):
    print(f"Running {placement_class.__name__}...")
    
    # Create placement instance
    placer = placement_class()
    
    # Generate allocation
    allocation = placer.generate_allocation(topology, applications, users)
    
    # Save to JSON
    alloc_dict = {"initialAllocation": allocation}
    
    with open(output_file, 'w') as f:
        json.dump(alloc_dict, f, indent=2)
    
    print(f"  - Generated {len(allocation)} module allocations")
    print(f"Saved: {output_file}")


def main():    
    # Load scenario
    print("\nLoading scenario files...")
    topology, applications, users = load_scenario()
    print(f"  - Topology: {len(topology['entity'])} nodes, {len(topology['link'])} links")
    print(f"  - Applications: {len(applications)} apps")
    print(f"  - Users: {len(users['sources'])} sources")
    
    # Output directory
    scenarios_dir = Path(__file__).parent.parent / "scenarios"
    
    # Paper algorithms — Pakpahan et al. (2025)
    for name, cls in [
        ("RDM",  RDMPlacement),
        ("SM",   SMPlacement),
        ("FFHA", FFHAPlacement),
        ("Hop2", Hop2Placement),
        ("Hop3", Hop3Placement),
        ("FFF",  FFFPlacement),
        ("GA",   GAPlacement),
        # Additional algorithms (lightweight implementations)
        ("ACO",   ACOPlacement),
        ("SA",    SAPlacement),
        ("TS",    TSPlacement),
        ("GWO",   GWOPlacement),
        ("WOA",   WOAPlacement),
        ("NSGA2", NSGAIIPlacement),
        ("MOEAD", MOEADPlacement),
    ]:
        print("\n" + "-" * 60)
        out = scenarios_dir / f"allocDefinition{name}.json"
        generate_placement(cls, topology, applications, users, out)
    
    print("\n" + "=" * 60)
    print("Placement generation complete!")
    print("=" * 60)
    
    
    print("\nNext step: Run runner/run_simulation.py to execute simulations.")


if __name__ == "__main__":
    main()
