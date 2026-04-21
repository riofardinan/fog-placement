"""
Placement algorithms for fog computing simulation.
"""

# Paper algorithms — Pakpahan et al. (2025)
from .rdm_placement import RDMPlacement
from .sm_placement import SMPlacement          # SortMatch
from .ffha_placement import FFHAPlacement      # FirstFitHopAware
from .hop2_placement import Hop2Placement
from .hop3_placement import Hop3Placement
from .fff_placement import FFFPlacement        # FrameworkFirstFit
from .ga_placement import GAPlacement

# Other algorithms (existing)
from .gr_placement import GRPlacement          # Greedy (Closest Resource First)
from .cn_placement import CNPlacement
from .ilp_placement import ILPPlacement
from .pso_placement import PSOPlacement
from .cngapso_placement import CNGAPSOPlacement
from .rl_placement import RLPlacement
from .gnn_placement import GNNPlacement
from .placement import Placement

__all__ = [
    # Paper algorithms
    "RDMPlacement",
    "SMPlacement",
    "FFHAPlacement",
    "Hop2Placement",
    "Hop3Placement",
    "FFFPlacement",
    "GAPlacement",
    # Other
    "GRPlacement",
    "CNPlacement",
    "ILPPlacement",
    "PSOPlacement",
    "CNGAPSOPlacement",
    "RLPlacement",
    "GNNPlacement",
]
