"""Pixel-area quadrature of normalized reciprocal peaks on the Ewald sphere.

The detector is a centered plane x=D and the incident beam is +x. Reflection
support is intersected with the outgoing sphere, then conservatively bounded
on that plane. Peak centers outside the detector are intentionally retained if
their support overlaps native pixels. Intensities are never renormalized at an
edge. All arrays are in native row/column convention.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .reciprocal_kernel import ewald_sphere_integral, gaussian_reciprocal_density


@dataclass(frozen=True)
class PlanarDetector:
    shape: tuple[int,int] = (512,512)
    pixel_pitch_mm: float = .1
    distance_mm: float = 60.
    beam_center_row_col: tuple[float,float] | None = None

    def __post_init__(self):
        shape = tuple(self.shape)
        if len(shape) != 2 or any(isinstance(v,bool) or not isinstance(v,(int,np.integer)) or not 1 <= v <= 4096 for v in shape):
            raise ValueError("Detector shape must be two positive integer dimensions <=4096.")
        if not np.all(np.isfinite([self.pixel_pitch_mm,self.distance_mm])) or min(self.pixel_pitch_mm,self.distance_mm) <= 0:
            raise ValueError("Positive finite detector pitch/distance are required.")
        center = tuple((v-1)/2 for v in shape) if self.beam_center_row_col is None else tuple(self.beam_center_row_col)
        if len(center) != 2 or not np.all(np.isfinite(center)):
            raise ValueError("Beam center must be finite native row/column coordinates.")
        object.__setattr__(self,"shape",shape)
        object.__setattr__(self,"beam_center_row_col",center)

    def to_dict(self):
        return {"shape":list(self.shape),"pixel_pitch_mm":float(self.pixel_pitch_mm),
                "distance_mm":float(self.distance_mm),"beam_center_row_col":list(self.beam_center_row_col)}

    def sample_rays(self, rows, cols, *, row_offset=0., col_offset=0., wavelength_A):
        """G(ray), pixel-center differential solid angle, and outgoing unit ray."""
        row, col = np.broadcast_arrays(np.asarray(rows,dtype=float),np.asarray(cols,dtype=float))
        y = (col+col_offset-self.beam_center_row_col[1])*self.pixel_pitch_mm
        z = (row+row_offset-self.beam_center_row_col[0])*self.pixel_pitch_mm
        length = np.sqrt(self.distance_mm**2+y*y+z*z)
        direction = np.stack((np.full_like(y,self.distance_mm),y,z),axis=-1)/length[...,None]
        reciprocal = (direction-np.array([1.,0.,0.]))/wavelength_A
        solid_angle = self.distance_mm*self.pixel_pitch_mm**2/length**3
        return reciprocal, solid_angle, direction


@dataclass(frozen=True)
class ProjectionResult:
    image: np.ndarray
    candidate_count: int
    rendered_count: int
    omitted_support_rate_bound: float
    support_sigma: float
    pixel_subdivision: int
    minimum_width_sampling_ratio: float = float("inf")
    undersampling_waived: bool = False


def _coordinate_range(n0, cosine, sine, axis):
    center = n0[axis]
    spread = np.sqrt(max(0.,1-center*center))*sine
    low,high = center*cosine-spread,center*cosine+spread
    if center >= cosine: high = 1.
    if -center >= cosine: low = -1.
    return low,high


def _patch(center, sigma, wavelength, detector, support_sigma):
    """Conservative detector rectangle of a reciprocal-ball/sphere intersection."""
    k = 1./wavelength
    outgoing = center+np.array([k,0.,0.])
    length = np.linalg.norm(outgoing)
    radius = support_sigma*sigma
    if abs(length-k) > radius: return None
    if length == 0 or radius >= length+k:
        return (0,detector.shape[0],0,detector.shape[1])
    cosine = np.clip((k*k+length*length-radius*radius)/(2*k*length),-1.,1.)
    sine = np.sqrt(max(0.,1-cosine*cosine))
    n0 = outgoing/length
    xmin,xmax = _coordinate_range(n0,cosine,sine,0)
    if xmax <= 0: return None
    if xmin <= 1e-12:
        return (0,detector.shape[0],0,detector.shape[1])
    ranges=[]
    for axis in [2,1]:
        low,high = _coordinate_range(n0,cosine,sine,axis)
        corners=[low/xmin,low/xmax,high/xmin,high/xmax]
        scale=detector.distance_mm/detector.pixel_pitch_mm
        index=0 if axis==2 else 1
        lower=min(corners)*scale+detector.beam_center_row_col[index]
        upper=max(corners)*scale+detector.beam_center_row_col[index]
        # Include every pixel whose area, not merely its center, intersects.
        start=max(0,min(detector.shape[index],math.floor(lower-.5)))
        stop=max(0,min(detector.shape[index],math.ceil(upper+.5)+1))
        ranges.append((start,stop))
    (r0,r1),(c0,c1)=ranges
    return None if r1<=r0 or c1<=c0 else (r0,r1,c0,c1)


def render_reciprocal_peaks(centers_G_A_inv, strengths, sigma_G_A_inv, *,
                            wavelength_A, detector=PlanarDetector(), support_sigma=6.,
                            pixel_subdivision=2, polarization="unpolarized",
                            max_working_bytes=256*1024**2, allow_undersampled=False):
    """Integrate peaks into native events/s after polarization and acceptance.

    strengths multiply K directly, i.e. include incident flux density,
    classical-electron-radius squared, illuminated volume/cell-volume squared,
    |F|^2 and any per-grain population weights exactly once. No Lorentz or
    separate excitation weight is applied here. Polarization may be None,
    'unpolarized', 'linear_y', or 'linear_z'. The returned omission bound is a
    complete-sphere upper bound; pixel quadrature error requires refinement.
    ``pixel_subdivision`` specifies Gauss-Legendre nodes per pixel axis.
    A conservative width-resolution guard rejects underresolved kernels before
    evaluating their samples. ``allow_undersampled=True`` is solely a diagnostic
    escape hatch, recorded in the result, never a convergence certificate.
    """
    centers=np.asarray(centers_G_A_inv,dtype=float)
    coefficients=np.asarray(strengths,dtype=float)
    if centers.ndim!=2 or centers.shape[1]!=3 or coefficients.shape!=(len(centers),):
        raise ValueError("Require [N,3] centers and [N] strengths.")
    sigmas=np.broadcast_to(np.asarray(sigma_G_A_inv,dtype=float),(len(centers),))
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(coefficients)) or np.any(coefficients<0) or not np.all(np.isfinite(sigmas)) or np.any(sigmas<=0):
        raise ValueError("Finite centers/nonnegative strengths/positive widths are required.")
    if not np.isfinite(wavelength_A) or wavelength_A<=0 or not np.isfinite(support_sigma) or support_sigma<=0:
        raise ValueError("Positive wavelength and support radius are required.")
    if isinstance(pixel_subdivision,bool) or not isinstance(pixel_subdivision,(int,np.integer)) or not 1<=pixel_subdivision<=32:
        raise ValueError("pixel_subdivision must be an integer in [1,32].")
    if polarization not in (None,"unpolarized","linear_y","linear_z"):
        raise ValueError("Unsupported polarization state.")
    if isinstance(max_working_bytes,bool) or not isinstance(max_working_bytes,(int,np.integer)) or max_working_bytes<1:
        raise ValueError("Positive working-memory budget is required.")
    if not isinstance(allow_undersampled,bool):
        raise ValueError("allow_undersampled must be boolean.")
    positive=coefficients>0
    step_bound=detector.pixel_pitch_mm/(detector.distance_mm*wavelength_A*pixel_subdivision)
    sampling_ratio=float(np.min(sigmas[positive])/step_bound) if np.any(positive) else float("inf")
    if sampling_ratio < 1. and not allow_undersampled:
        required=math.ceil(detector.pixel_pitch_mm/(detector.distance_mm*wavelength_A*np.min(sigmas[positive])))
        raise ValueError(f"Pixel quadrature undersamples reciprocal width; require at least {required} nodes/axis or a finer detector grid.")
    output_bytes=int(np.prod(detector.shape))*8
    # Complete-cloud analytical support accounting uses temporary vector arrays;
    # final immutable bytes-backed output temporarily coexists with the buffer.
    if 2*output_bytes+160*len(centers)+4096>max_working_bytes:
        raise MemoryError("Native output exceeds projector working-memory budget.")
    image=np.zeros(detector.shape,dtype=float)
    rendered=0
    omissions=ewald_sphere_integral(centers,sigmas,wavelength_A,outside_radius=support_sigma*sigmas)
    omission_bound=float(np.dot(coefficients,omissions))
    del omissions
    outgoing_length=np.linalg.norm(centers+np.array([1/wavelength_A,0.,0.]),axis=1)
    selected=(coefficients>0)&(np.abs(outgoing_length-1/wavelength_A)<=support_sigma*sigmas)
    candidates=int(np.count_nonzero(selected))
    del outgoing_length
    nodes,weights=np.polynomial.legendre.leggauss(pixel_subdivision)
    offsets=nodes*.5
    weights=weights*.5
    per_point_bytes=192  # conservative geometric/kernel scratch, not OS RSS
    for center,coefficient,sigma in zip(centers[selected],coefficients[selected],sigmas[selected]):
        patch=_patch(center,sigma,wavelength_A,detector,support_sigma)
        if patch is None: continue
        r0,r1,c0,c1=patch
        rows_per_chunk=max(1,(max_working_bytes-output_bytes)//max(per_point_bytes*(c1-c0),1))
        if per_point_bytes*(c1-c0)>max_working_bytes-output_bytes:
            raise MemoryError("A detector row exceeds projector scratch-memory budget.")
        nonzero=False
        for first in range(r0,r1,rows_per_chunk):
            stop=min(r1,first+rows_per_chunk)
            row,col=np.meshgrid(np.arange(first,stop),np.arange(c0,c1),indexing="ij")
            value=np.zeros(row.shape,dtype=float)
            for ro,rw in zip(offsets,weights):
                for co,cw in zip(offsets,weights):
                    g,solid,direction=detector.sample_rays(row,col,row_offset=ro,col_offset=co,wavelength_A=wavelength_A)
                    delta=g-center
                    squared=np.sum(delta*delta,axis=-1)
                    density=gaussian_reciprocal_density(delta,sigma)
                    density[squared>(support_sigma*sigma)**2]=0.
                    if polarization=="unpolarized": factor=(1+direction[...,0]**2)/2
                    elif polarization=="linear_y": factor=1-direction[...,1]**2
                    elif polarization=="linear_z": factor=1-direction[...,2]**2
                    else: factor=1.
                    value+=rw*cw*density*solid*factor
            value*=coefficient
            image[first:stop,c0:c1]+=value
            nonzero=nonzero or bool(np.any(value>0))
            del row,col,value,g,solid,direction,delta,squared,density,factor
        rendered+=int(nonzero)
    if not np.all(np.isfinite(image)) or not np.isfinite(omission_bound):
        raise FloatingPointError("Reciprocal projection overflowed; revise physical/numerical scales.")
    image=np.frombuffer(image.tobytes(),dtype=np.float64).reshape(image.shape)
    return ProjectionResult(image,candidates,rendered,omission_bound,float(support_sigma),int(pixel_subdivision),
                            sampling_ratio,bool(allow_undersampled and sampling_ratio<1.))
