"""GeoP2E geometric primitives — math kernel (pure numpy/scipy, open3d-free).

Ported from the project notebooks (`feature_Registration_core_data.ipynb`,
`totalphase3.ipynb`) with two corrections/upgrades relative to the submitted
manuscript + original code:

1. LEFT-perturbation convention, consistent with `UNIFIED_MATH.md`:
   the Jacobian is evaluated at the *scan-frame* transformed point x_i = R p_i + t
   (the original code already did this; the manuscript's Eq.(11) used p_i by typo).
2. ANALYTICAL cylinder Jacobian added. The original notebook's
   `CylinderFeature.compute_jacobian` fell back to a numerical central
   difference; the submitted paper claims a unified analytical Jacobian for all
   three primitives. We now provide the true analytical form (identical
   structure to plane/sphere, J_i = [(x_i x n_i), n_i]), so the code matches the
   claim. The numerical Jacobian is retained as a toggle for the
   analytic-vs-numerical ablation (reviewer R1#5 / R3).

All primitives are exact signed-distance residuals (eikonal gradient = unit
normal n_i); see `UNIFIED_MATH.md`.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rot


class GeometricFeature(ABC):
    """Base class for an explicit parametric CAD primitive."""

    kind: str = "generic"

    # ---- core interface -------------------------------------------------
    @abstractmethod
    def compute_residual(self, points: np.ndarray) -> np.ndarray:
        """Exact signed-distance residual r_i = phi(x_i), shape (N,)."""

    @abstractmethod
    def compute_feature_normal(self, points: np.ndarray) -> np.ndarray:
        """Unit surface normal n_i at each point (outward radial / plane n)."""

    @abstractmethod
    def generate_points(self, num_points: int, noise_std: float = 0.0,
                        extent: float = 10.0, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample points exactly on the primitive (for synthetic GT)."""

    # ---- Jacobians ------------------------------------------------------
    def compute_jacobian(self, points: np.ndarray) -> np.ndarray:
        """Analytical residual Jacobian w.r.t. LEFT se(3) perturbation xi=[w,v].

        Unified form  J_i = [ (x_i x n_i)^T , n_i^T ] in R^{1x6}  (stacked N x 6),
        derived from the eikonal identity g_i = n_i and dx_i = w x x_i + v.
        """
        n = self.compute_feature_normal(points)              # (N,3)
        J = np.empty((points.shape[0], 6), dtype=points.dtype)
        J[:, :3] = np.cross(points, n)                       # rotational block
        J[:, 3:] = n                                         # translational block
        return J

    def compute_numerical_jacobian(self, points: np.ndarray, delta: float = 1e-6) -> np.ndarray:
        """Central-difference Jacobian: perturbs the scan-frame point on the LEFT,
        x' = exp(w^) x_i + v, matching the analytical convention. Used as a
        fallback and as the 'numerical Jacobian' ablation variant.
        """
        N = points.shape[0]
        J = np.zeros((N, 6))
        for i in range(6):
            perturb = np.zeros(6)
            perturb[i] = delta
            p_pos = points @ Rot.from_rotvec(perturb[:3]).as_matrix().T + perturb[3:]
            p_neg = points @ Rot.from_rotvec(-perturb[:3]).as_matrix().T - perturb[3:]
            J[:, i] = (self.compute_residual(p_pos) - self.compute_residual(p_neg)) / (2.0 * delta)
        return J

    # ---- helpers --------------------------------------------------------
    @staticmethod
    def orthogonal_basis(vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vec = np.asarray(vec, dtype=np.float64)
        vec /= np.linalg.norm(vec)
        arbitrary = np.array([0.0, 1.0, 0.0]) if abs(vec[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(vec, arbitrary); u /= np.linalg.norm(u)
        v = np.cross(vec, u)
        return u, v


class PlaneFeature(GeometricFeature):
    kind = "plane"

    def __init__(self, normal, d: float):
        n = np.asarray(normal, dtype=np.float64)
        self.normal = n / np.linalg.norm(n)
        self.d = float(d)

    def compute_residual(self, points):
        return points @ self.normal + self.d

    def compute_feature_normal(self, points):
        return np.tile(self.normal, (points.shape[0], 1))

    def generate_points(self, num_points, noise_std=0.0, extent=5.0, rng=None):
        rng = rng or np.random.default_rng()
        c = -self.d * self.normal
        u, v = self.orthogonal_basis(self.normal)
        a = rng.uniform(-extent, extent, num_points)
        b = rng.uniform(-extent, extent, num_points)
        pts = c + np.outer(a, u) + np.outer(b, v)
        if noise_std > 0:
            pts += rng.normal(0, noise_std, pts.shape)
        return pts


class CylinderFeature(GeometricFeature):
    kind = "cylinder"

    def __init__(self, center, axis, radius: float):
        self.center = np.asarray(center, dtype=np.float64)
        a = np.asarray(axis, dtype=np.float64); self.axis = a / np.linalg.norm(a)
        self.radius = float(radius)
        self._Pperp = np.eye(3) - np.outer(self.axis, self.axis)  # projector ||-> perp

    def compute_residual(self, points):
        # exact signed radial distance ||P_perp(x - q)|| - r
        return np.linalg.norm((points - self.center) @ self._Pperp.T, axis=1) - self.radius

    def compute_feature_normal(self, points):
        radial = (points - self.center) @ self._Pperp.T
        norms = np.linalg.norm(radial, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)   # off-medial-axis guard
        return radial / norms

    def generate_points(self, num_points, noise_std=0.0, extent=10.0, rng=None):
        rng = rng or np.random.default_rng()
        u, v = self.orthogonal_basis(self.axis)
        h = rng.uniform(-extent / 2, extent / 2, num_points)
        th = rng.uniform(0, 2 * np.pi, num_points)
        pts = (self.center + np.outer(h, self.axis)
               + self.radius * (np.outer(np.cos(th), u) + np.outer(np.sin(th), v)))
        if noise_std > 0:
            pts += rng.normal(0, noise_std, pts.shape)
        return pts


class SphereFeature(GeometricFeature):
    kind = "sphere"

    def __init__(self, center, radius: float):
        self.center = np.asarray(center, dtype=np.float64)
        self.radius = float(radius)

    def compute_residual(self, points):
        return np.linalg.norm(points - self.center, axis=1) - self.radius

    def compute_feature_normal(self, points):
        vec = points - self.center
        d = np.linalg.norm(vec, axis=1, keepdims=True)
        d = np.where(d < 1e-9, 1.0, d)
        return vec / d

    def generate_points(self, num_points, noise_std=0.0, extent=0.0, rng=None):
        rng = rng or np.random.default_rng()
        v = rng.normal(0, 1, (num_points, 3))
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts = self.center + v * self.radius
        if noise_std > 0:
            pts += rng.normal(0, noise_std, pts.shape)
        return pts


def load_features_from_json(path: str):
    """Load Plane/Cylinder/Sphere features from refined_features.json
    (keys: Plane[{coefficients:[a,b,c,d]}], Cylinder[{location,z_axis,radius}],
    Sphere[{location,radius}])."""
    import json, os
    if not os.path.exists(path):
        return []
    data = json.load(open(path))
    feats = []
    for f in data.get("Plane", []):
        c = np.asarray(f["coefficients"], dtype=np.float64)
        nrm = np.linalg.norm(c[:3])
        feats.append(PlaneFeature(c[:3] / nrm, c[3] / nrm) if nrm > 1e-9 else PlaneFeature(c[:3], c[3]))
    for f in data.get("Cylinder", []):
        feats.append(CylinderFeature(f["location"], f["z_axis"], f["radius"]))
    for f in data.get("Sphere", []):
        feats.append(SphereFeature(f["location"], f["radius"]))
    return feats
