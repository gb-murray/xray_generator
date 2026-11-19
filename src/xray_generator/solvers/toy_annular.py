"""Fixed-geometry Gaussian/von-Mises renderer for synthetic ring fixtures.

The templates describe synthetic angular profiles rather than crystalline phases.
Budgets are defined before analysis masks and counting noise.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import i0e

from .contracts import readonly_array

TEMPLATES = {
    "A": ([1.30,1.82,2.35,2.96,3.62,4.20],[1.,.65,.45,.35,.25,.20]),
    "B": ([1.44,1.83,2.58,3.12,3.65,4.34],[1.,.55,.70,.25,.40,.20]),
    "C": ([1.58,2.05,2.36,3.13,3.84,4.28],[.70,1.,.45,.60,.25,.35]),
    "D": ([1.31,2.19,2.59,3.42,3.86,4.35],[.75,.55,1.,.45,.30,.20]),
}


@dataclass(frozen=True)
class PilotGeometry:
    shape: tuple[int,int] = (512,512)
    pixel_pitch_mm: float = .1
    distance_mm: float = 60.
    wavelength_A: float = .5166
    q_range_A_inv: tuple[float,float] = (1.,4.5)

    def coordinates(self):
        if len(self.shape)!=2 or any(isinstance(v,bool) or int(v)!=v or not 2<=v<=2048 for v in self.shape):
            raise ValueError("Invalid native detector shape.")
        if not np.all(np.isfinite([self.pixel_pitch_mm,self.distance_mm,self.wavelength_A,*self.q_range_A_inv])):
            raise ValueError("Nonfinite detector parameters.")
        if min(self.pixel_pitch_mm,self.distance_mm,self.wavelength_A)<=0 or not 0<=self.q_range_A_inv[0]<self.q_range_A_inv[1]:
            raise ValueError("Invalid positive geometry or q range.")
        row,col=np.indices(self.shape,dtype=float)
        y=(col-(self.shape[1]-1)/2)*self.pixel_pitch_mm
        z=(row-(self.shape[0]-1)/2)*self.pixel_pitch_mm
        radius=np.hypot(y,z)
        q=4*np.pi*np.sin(.5*np.arctan2(radius,self.distance_mm))/self.wavelength_A
        chi=np.mod(np.arctan2(z,y),2*np.pi)
        normalizer=(self.distance_mm/np.sqrt(self.distance_mm**2+radius**2))**3
        return q,chi,normalizer


@dataclass(frozen=True)
class AngularProfile:
    family: str = "uniform"
    anisotropic_fraction: float = 0.
    kappa: float = 8.
    orientation_rad: float = 0.
    second_separation_rad: float = np.pi/2
    first_lobe_weight: float = .5

    def evaluate(self,chi):
        params=[self.anisotropic_fraction,self.kappa,self.orientation_rad,
                self.second_separation_rad,self.first_lobe_weight]
        if not np.all(np.isfinite(params)) or not 0<=self.anisotropic_fraction<=1 or not 0<=self.first_lobe_weight<=1 or self.kappa<0:
            raise ValueError("Invalid angular profile parameters.")
        if self.family=="uniform": return np.full_like(chi,1/(2*np.pi))
        def vm(phi):
            return np.exp(self.kappa*(np.cos(chi-phi)-1))/(2*np.pi*i0e(self.kappa))
        first=vm(self.orientation_rad)
        if self.family=="single_von_mises": shaped=first
        elif self.family=="antipodal_equal_von_mises": shaped=.5*(first+vm(self.orientation_rad+np.pi))
        elif self.family=="two_von_mises":
            shaped=self.first_lobe_weight*first+(1-self.first_lobe_weight)*vm(self.orientation_rad+self.second_separation_rad)
        else: raise ValueError("Unknown controlled angular family.")
        return (1-self.anisotropic_fraction)/(2*np.pi)+self.anisotropic_fraction*shaped


def render_expected(*, components=("A","B"), weights=(.5,.5), profiles=None,
                    radial_sigma_A_inv=.05, bragg_events=1024.,background_events=1024.,
                    geometry=PilotGeometry()):
    weights=np.asarray(weights,dtype=float)
    if len(components)!=len(weights) or not len(weights) or len(set(components))!=len(components):
        raise ValueError("Unique components and weights must match.")
    if not np.all(np.isfinite(weights)) or np.any(weights<0) or not np.isclose(weights.sum(),1.,atol=1e-12,rtol=0):
        raise ValueError("Synthetic photon fractions must be a simplex.")
    if not np.isfinite(radial_sigma_A_inv) or radial_sigma_A_inv<=0:
        raise ValueError("Positive radial sigma is required.")
    if not np.all(np.isfinite([bragg_events,background_events])) or min(bragg_events,background_events)<0:
        raise ValueError("Nonnegative finite event budgets are required.")
    profiles=tuple(profiles) if profiles is not None else (AngularProfile(),)*len(weights)
    if len(profiles)!=len(weights): raise ValueError("Each component requires an angular profile.")
    q,chi,normalizer=geometry.coordinates()
    roi=(q>=geometry.q_range_A_inv[0])&(q<=geometry.q_range_A_inv[1])
    signal=np.zeros(geometry.shape)
    for name,weight,profile in zip(components,weights,profiles):
        if name not in TEMPLATES: raise ValueError("Unknown artificial component.")
        centers,amplitudes=TEMPLATES[name]
        radial=sum(a*np.exp(-.5*((q-c)/radial_sigma_A_inv)**2) for c,a in zip(centers,amplitudes))
        clean=radial*profile.evaluate(chi)*normalizer*roi
        if clean.sum()<=0: raise ValueError("Component has no support in the native ROI.")
        signal+=bragg_events*weight*clean/clean.sum()
    background=normalizer*roi
    if background.sum()<=0: raise ValueError("No native q coverage.")
    background=background_events*background/background.sum()
    return {"signal_mean":readonly_array(signal),"background_mean":readonly_array(background),
            "total_mean":readonly_array(signal+background),"q_A_inv":readonly_array(q),
            "chi_rad":readonly_array(chi),"normalizer":readonly_array(normalizer),
            "roi":roi,"scale_origin":"synthetic_budget","profile":"controlled_annular_renderer_fixture_v1"}
