"""Persistent volume-weighted orientation populations using existing ODF math.

These populations supply crystallographic orientations, not independent angular
profiles pasted onto rings. They imply no positioned grain geometry. The domain
width and illuminated material amount belong to the scattering backend.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np

from src.classes_and_functions.orientation import quaternion_to_rotation_matrix
from src.classes_and_functions.texture import odf_from_dict

from ..experiment.provenance import fingerprint
from ..experiment.random import keyed_rng
from ..model.conventions import frozen_array


def _unit_tuple(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.max(np.abs(array)) == 0:
        raise ValueError(f"{name} requires a finite nonzero {size}-vector.")
    array = array/np.max(np.abs(array))
    array = array/np.linalg.norm(array)
    if size == 4 and array[0] < 0:
        array = -array
    return tuple(float(v) for v in array)


@dataclass(frozen=True)
class TextureConfig:
    """Physical ODF recipe; directions are Cartesian crystal/sample vectors.

    They are not Miller-index triples for a general skew lattice. Convert a
    lattice direction through that lattice's direct metric before supplying it.
    ``sample_to_lab_wxyz`` is an active, fixed acquisition pose.
    """
    kind: str = "random"
    num_orientations: int = 1024
    seed: int = 12345
    crystal_direction: tuple[float,float,float] = (0.,0.,1.)
    sample_direction: tuple[float,float,float] = (0.,0.,1.)
    spread_deg: float = 10.
    center_quaternion: tuple[float,float,float,float] = (1.,0.,0.,0.)
    spin_center_deg: float = 0.
    spin_width_deg: float = 360.
    spin_kappa: float = 0.
    isotropic_fraction: float = 0.
    sample_to_lab_wxyz: tuple[float,float,float,float] = (1.,0.,0.,0.)

    def __post_init__(self):
        if self.kind not in {"random", "fiber", "partial_fiber", "component"}:
            raise ValueError("Unsupported physical orientation-population family.")
        if isinstance(self.num_orientations,bool) or not isinstance(self.num_orientations,int) or not 1 <= self.num_orientations <= 1000000:
            raise ValueError("num_orientations must be a bounded positive integer.")
        if isinstance(self.seed,bool) or not isinstance(self.seed,int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if not all(math.isfinite(v) for v in [self.spread_deg,self.spin_center_deg,self.spin_width_deg,self.spin_kappa,self.isotropic_fraction]):
            raise ValueError("Texture parameters must be finite.")
        if not 0 < self.spread_deg <= 180 or not 0 < self.spin_width_deg <= 360 or self.spin_kappa < 0 or not 0 <= self.isotropic_fraction <= 1:
            raise ValueError("Texture parameters are outside their supported range.")
        for field,size in [("crystal_direction",3),("sample_direction",3),("center_quaternion",4),("sample_to_lab_wxyz",4)]:
            object.__setattr__(self,field,_unit_tuple(getattr(self,field),size,field))

    def odf_dict(self):
        if self.kind == "random" or self.isotropic_fraction == 1:
            shape = {"type":"random"}
        elif self.kind == "component":
            shape = {"type":"component", "center_quaternion":list(self.center_quaternion), "spread_deg":self.spread_deg}
        else:
            shape = {"type":self.kind, "crystal_direction":list(self.crystal_direction),
                     "sample_direction":list(self.sample_direction), "spread_deg":self.spread_deg}
            if self.kind == "partial_fiber":
                shape.update(spin_center_deg=self.spin_center_deg,spin_width_deg=self.spin_width_deg,spin_kappa=self.spin_kappa)
        if self.kind != "random" and 0 < self.isotropic_fraction < 1:
            return {"type":"mixture", "components":[dict(shape,weight=1-self.isotropic_fraction),
                                                      {"type":"random","weight":self.isotropic_fraction}]}
        return shape

    def to_dict(self):
        value = asdict(self)
        for key in ["crystal_direction","sample_direction","center_quaternion","sample_to_lab_wxyz"]:
            value[key] = list(value[key])
        return value

    @classmethod
    def from_dict(cls,value):
        return cls(**value)


@dataclass(frozen=True)
class OrientationEnsemble:
    quaternions_wxyz: np.ndarray
    volume_weights: np.ndarray
    config: TextureConfig

    def __post_init__(self):
        q = np.asarray(self.quaternions_wxyz, dtype=float)
        weights = np.asarray(self.volume_weights,dtype=float)
        if q.shape != (self.config.num_orientations,4) or weights.shape != (len(q),):
            raise ValueError("Orientation/weight shapes do not match the declared population.")
        if not np.all(np.isfinite(q)) or not np.allclose(np.linalg.norm(q,axis=1),1,rtol=0,atol=1e-12):
            raise ValueError("Population quaternions must be finite unit orientations.")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.max() <= 0:
            raise ValueError("Volume weights must be finite, nonnegative and nonzero.")
        # Scaling before summation avoids overflow for valid large volumes.
        scaled_weights = weights / weights.max()
        object.__setattr__(self,"quaternions_wxyz",frozen_array(q))
        object.__setattr__(self,"volume_weights",frozen_array(scaled_weights/scaled_weights.sum()))

    @property
    def rotation_matrices(self):
        """Active crystal-to-sample matrices."""
        return frozen_array(quaternion_to_rotation_matrix(self.quaternions_wxyz))

    @property
    def lab_rotation_matrices(self):
        """Active crystal-to-lab matrices, with sample pose applied exactly once."""
        pose=quaternion_to_rotation_matrix(np.asarray(self.config.sample_to_lab_wxyz))
        return frozen_array(np.einsum("ij,njk->nik",pose,self.rotation_matrices))

    @property
    def fingerprint(self):
        return fingerprint({"implementation":"legacy_odf_volume_population_v1", "config":self.config.to_dict(),
                            "quaternions":self.quaternions_wxyz,"volume_weights":self.volume_weights})


def orientation_ensemble(config: TextureConfig, volume_weights=None):
    if not isinstance(config,TextureConfig):
        raise TypeError("config must be TextureConfig.")
    odf = config.odf_dict()
    # Sample pose is an acquisition parameter: changing it must rotate, rather
    # than redraw, the same specimen orientation population.
    rng=keyed_rng(config.seed,"m3b_texture_population",{"odf":odf,"num_orientations":config.num_orientations})
    if odf["type"] == "mixture":
        # The legacy generic mixture sampler invokes empty component batches.
        # Some physical ODFs do not define sample(0), so skip empty selections.
        components = odf["components"]
        choices = rng.choice(len(components),config.num_orientations,p=[item["weight"] for item in components])
        q = np.empty((config.num_orientations,4))
        for index, item in enumerate(components):
            mask = choices == index
            if np.any(mask):
                q[mask] = odf_from_dict(item).sample(int(mask.sum()),rng)
    else:
        q=odf_from_dict(odf).sample(config.num_orientations,rng)
    weights=np.ones(config.num_orientations) if volume_weights is None else volume_weights
    return OrientationEnsemble(q,weights,config)
