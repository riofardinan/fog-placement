"""
Multi-objective placement algorithms.

These implementations keep code intentionally lightweight:
- objectives are computed from topology hops + RAM validity
- selection uses simple Pareto ranking/crowding approximations
"""

from .nsga2_placement import NSGAIIPlacement
from .moead_placement import MOEADPlacement

__all__ = ["NSGAIIPlacement", "MOEADPlacement"]

