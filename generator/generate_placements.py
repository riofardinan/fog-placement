"""
Placement Generator for YAFS 3.1
"""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Paper algorithms — Pakpahan et al. (2025)
from placements.rdm_placement import RDMPlacement
from placements.sm_placement import SMPlacement      # SortMatch
from placements.ffha_placement import FFHAPlacement  # FirstFitHopAware
from placements.hop2_placement import Hop2Placement
from placements.hop3_placement import Hop3Placement
from placements.fff_placement import FFFPlacement    # FrameworkFirstFit
from placements.ga_placement import GAPlacement
# Other algorithms
from placements.gr_placement import GRPlacement      # Greedy
from placements.cn_placement import CNPlacement
from placements.ilp_placement import ILPPlacement
from placements.pso_placement import PSOPlacement
from placements.cngapso_placement import CNGAPSOPlacement


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
    """
    Generate allocation using a placement algorithm.
    
    Args:
        placement_class: Placement algorithm class
        topology: Topology dict
        applications: Applications list
        users: Users dict
        output_file: Output JSON file path
    """
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
    print(f"✓ Saved: {output_file}")


def main():
    """Main placement generator function."""
    print("=" * 60)
    print("YAFS Placement Generator")
    print("=" * 60)
    
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
