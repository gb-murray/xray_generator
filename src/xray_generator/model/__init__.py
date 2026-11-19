"""Immutable contracts for the compact forward-model path."""

from .materials import PhaseDefinition
from .sample import FractionRegion, MaterialRecipe, MaterialStateBatch

__all__ = ["PhaseDefinition", "FractionRegion", "MaterialRecipe", "MaterialStateBatch"]
