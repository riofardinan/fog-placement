"""
Run all placement algorithms sequentially.
Convenient script to execute all three placements in one go.
"""
from run_simulation import run_simulation


def main():
    """Run simulations for all placement algorithms."""
    placements = ["CNPlacement", "GAPlacement", "ILPPlacement", "RLPlacement", "GNNPlacement"]
    duration = 1000000
    
    print("\n" + "=" * 70)
    print("Running ALL Placement Simulations")
    print("=" * 70)
    
    results = {}
    
    for placement in placements:
        print(f"\n{'=' * 70}")
        print(f"[{placements.index(placement) + 1}/{len(placements)}] {placement}")
        print("=" * 70)
        
        try:
            results_dir = run_simulation(placement, duration)
            results[placement] = results_dir
            print(f"\n✓ {placement} completed successfully!")
        except Exception as e:
            print(f"\n❌ {placement} failed: {e}")
            results[placement] = None
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for placement, results_dir in results.items():
        if results_dir:
            print(f"✓ {placement}: {results_dir}")
        else:
            print(f"❌ {placement}: FAILED")
    
    print("\n" + "=" * 70)
    print("All simulations complete!")
    print("=" * 70)
    print("\nNext step: Analyze results in the results/ directory")


if __name__ == "__main__":
    main()
