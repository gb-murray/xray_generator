"""Procedural material evaluation and bounded geometry fixtures."""
from .procedural import ProceduralMaterial
from .lazy_grains import LazyGrainGenerator, GrainBatch

__all__ = ["ProceduralMaterial", "LazyGrainGenerator", "GrainBatch"]
