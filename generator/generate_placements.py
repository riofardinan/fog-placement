"""
Placement Generator for YAFS 3.1
Generates allocation (placement) configurations for different algorithms:
- CNPlacement (Complex Network based)
- GAPlacement (Genetic Algorithm)
- ILPPlacement (Integer Linear Programming)

This is part of RUN 1: Generate placement allocations before simulation.
"""
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from placements.cn_placement import CNPlacement
from placements.ga_placement import GAPlacement
from placements.ilp_placement import ILPPlacement
from placements.rl_placement import RLPlacement
from placements.gnn_placement import GNNPlacement


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
    
    # Generate CNPlacement allocation
    print("\n" + "-" * 60)
    cn_file = scenarios_dir / "allocDefinitionCN.json"
    generate_placement(CNPlacement, topology, applications, users, cn_file)
    
    # Generate GAPlacement allocation
    print("\n" + "-" * 60)
    ga_file = scenarios_dir / "allocDefinitionGA.json"
    generate_placement(GAPlacement, topology, applications, users, ga_file)
    
    # Generate ILPPlacement allocation
    print("\n" + "-" * 60)
    ilp_file = scenarios_dir / "allocDefinitionILP.json"
    generate_placement(ILPPlacement, topology, applications, users, ilp_file)
    
    # Generate RLPlacement allocation
    print("\n" + "-" * 60)
    rl_file = scenarios_dir / "allocDefinitionRL.json"
    generate_placement(RLPlacement, topology, applications, users, rl_file)
    
    # Generate GNNPlacement allocation
    print("\n" + "-" * 60)
    gnn_file = scenarios_dir / "allocDefinitionGNN.json"
    generate_placement(GNNPlacement, topology, applications, users, gnn_file)
    
    print("\n" + "=" * 60)
    print("Placement generation complete!")
    print("=" * 60)
    print("\nNext step: Run runner/run_simulation.py to execute simulations.")


if __name__ == "__main__":
    main()
