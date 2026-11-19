"""Normalized finite-domain scattering in crystallographic reciprocal units.

G and k use cycles/Angstrom, not radians/Angstrom. The Gaussian is a declared
finite-domain approximation to a shape transform. Its *three-dimensional*
integral is one. There is no separately peak-normalized rocking curve and no
normalization of a spot after detector clipping.
"""
from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def coherent_sigma_G_A_inv(domain_size_nm, shape_factor=.9):
    """Gaussian reciprocal sigma from Scherrer FWHM K/L; L is in nm."""
    length = np.asarray(domain_size_nm, dtype=np.float64)
    if np.any(~np.isfinite(length)) or np.any(length <= 0) or not np.isfinite(shape_factor) or shape_factor <= 0:
        raise ValueError("Coherent length and shape factor must be finite and positive.")
    return shape_factor / (10.0 * length * FWHM_TO_SIGMA)


def gaussian_reciprocal_density(delta_G_A_inv, sigma_G_A_inv):
    """Return normalized K(delta), in Angstrom^3, for [...,3] vectors."""
    delta = np.asarray(delta_G_A_inv, dtype=np.float64)
    sigma = np.asarray(sigma_G_A_inv, dtype=np.float64)
    if delta.shape[-1:] != (3,) or np.any(~np.isfinite(delta)):
        raise ValueError("Reciprocal differences must be finite [...,3] vectors.")
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Reciprocal sigma must be finite and positive.")
    squared = np.sum(delta * delta, axis=-1)
    return np.exp(-0.5*squared/sigma**2) / ((2*np.pi)**1.5 * sigma**3)


def powder_orientation_average(radius_G_A_inv, reflection_G_A_inv, sigma_G_A_inv):
    """Exact Haar average of one isotropic normalized reciprocal peak.

    Average over the direction of a reciprocal vector of fixed length g.
    The angular density is normalized over SO(3); no reflection multiplicity is
    supplied here. Each hkl in a reflection list contributes its own vector.
    """
    r, g, sigma = np.broadcast_arrays(np.asarray(radius_G_A_inv, dtype=float),
                                      np.asarray(reflection_G_A_inv, dtype=float),
                                      np.asarray(sigma_G_A_inv, dtype=float))
    if not np.all(np.isfinite([r,g,sigma])) or np.any(r < 0) or np.any(g < 0) or np.any(sigma <= 0):
        raise ValueError("Require nonnegative radii and positive finite sigma.")
    a = r*g/sigma**2
    angular = np.divide(-np.expm1(-2*a), 2*a, out=np.ones_like(a), where=a != 0)
    return np.exp(-.5*((r-g)/sigma)**2)*angular / ((2*np.pi)**1.5*sigma**3)


def ewald_sphere_integral(center_G_A_inv, sigma_G_A_inv, wavelength_A, *, outside_radius=None):
    """Integral of K over the complete outgoing-direction sphere in sr.

    Multiplying by a peak's coefficient bounds its accepted detector rate when
    polarization and detector efficiency lie in [0,1]. With outside_radius R,
    return only the contribution where |G(ray)-center| > R. This supplies a
    conservative absolute bound on reciprocal-support truncation, independently
    of whether a peak center lies within the detector.
    """
    centers = np.asarray(center_G_A_inv, dtype=np.float64)
    sigma = np.asarray(sigma_G_A_inv, dtype=np.float64)
    if centers.shape[-1:] != (3,) or np.any(~np.isfinite(centers)):
        raise ValueError("Centers require finite [...,3] reciprocal vectors.")
    if not np.isfinite(wavelength_A) or wavelength_A <= 0 or np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("Require positive finite sigma and wavelength.")
    k = 1.0/wavelength_A
    outgoing_center = centers + np.array([k,0.,0.])
    a_length = np.linalg.norm(outgoing_center,axis=-1)
    a_length, sigma = np.broadcast_arrays(a_length,sigma)
    minimum = np.abs(a_length-k)
    maximum = a_length+k
    if outside_radius is None:
        lower = minimum
    else:
        radius = np.asarray(outside_radius,dtype=float)
        if np.any(~np.isfinite(radius)) or np.any(radius < 0):
            raise ValueError("Support radius must be finite and nonnegative.")
        lower = np.maximum(minimum,radius)
    a = k*a_length/sigma**2
    span = np.maximum(0.,(maximum**2-lower**2)/(2*sigma**2))
    factor = np.divide(-np.expm1(-span),a,out=np.zeros_like(a),where=a != 0)
    normalizer = (2*np.pi)**-1.5*sigma**-3
    result = 2*np.pi*normalizer*np.exp(-.5*(lower/sigma)**2)*factor
    at_origin = 4*np.pi*normalizer*np.exp(-.5*(k/sigma)**2)
    if outside_radius is not None:
        at_origin = np.where(np.asarray(outside_radius) < k,at_origin,0.)
    return np.where(a == 0,at_origin,result)
