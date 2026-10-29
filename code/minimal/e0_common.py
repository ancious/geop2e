"""E0 common infrastructure for the minimal (reviewer-demands-only) experiment suite.

Shared by E2 (controlled Table A5 rebuild), E3 (HV with/without ablation), E4
(Cranfield non-gear study), E5 (fixed-scan repeatability, computed from E3's
real-scan full-HV arm).

Design per RINENG_EXPERIMENT_PLAN_MINIMAL.md (GPT-5.6-sol-max co-designed):
- one frozen configuration per method (published defaults; nothing tuned on the
  reported scans);
- identical inputs across methods: same scan files, same CAD version, same
  seeded target sampling, same evaluation mesh;
- simple, readable records — no hash/manifest machinery.

Runs in the `geo` env (py3.11 + open3d 0.19 + trimesh). Pure CPU.
"""
from __future__ import annotations
import os
import time
import json
import numpy as np
import open3d as o3d

GEAR = "data/gear_step_cad"
CRANFIELD = "data/cranfield"

MASTER_SEED = 20260827          # protocol v1.0-lite master seed
N_TARGET = 20000                # frozen CAD target sample size (matches prior runs)
TARGET_SAMPLING_SEED = 20260827 # o3d sample_points_uniformly is seeded via rng arg where supported

# The three defect scans named by both the local scripts and Table A5.
SCANS = [
    ("defect/22.pcd", "S1_defect_30pct"),
    ("defect/4.ply", "S2_defective"),
    ("defect/1.ply", "S3_fragment"),
]

# ---- frozen per-method configurations (published defaults; NOT tuned on the
#      three reported scans; any change here must be recorded in the results) ----
CONFIGS = {
    # shared FPFH/FGR coarse initialisation for all refinement-based methods
    "fgr_voxel": 1.0,                       # mm; ONE frozen setting (not 3 aggregated)
    "fgr_max_corr_dist": 1.5,               # mm
    "icp_threshold": 2.0,                   # mm, all open3d ICP variants
    "icp_max_iter": 100,
    "trimmed": {"threshold": 2.0, "trim_frac": 0.3, "max_iter": 50},
    "sparseicp": {"threshold": 3.0, "p": 0.5, "outer": 20},
    # GeoP2E solver params follow real_gear_experiment.auto_params (published);
    # they are scale-derived from the CAD, not per-scan tuned.
    "geop2e_select": {"criterion": "e", "target_size": 8, "min_features": 5},
    # TEASER++ : library defaults (noise_bound set from data scale, see e2)
    # Super4PCS: OpenGR CLI defaults (delta = 0.5 * mesh resolution proxy)
    # RPMNet  : bundled partial_0.7 checkpoint, 1024-pt input, zero-shot
}


# ---------------------------------------------------------------- data loading
def load_scan(path: str) -> o3d.geometry.PointCloud:
    """Robust scan loader (.pcd/.ply/.obj; falls back to trimesh vertices)."""
    if path.endswith(".obj"):
        m = o3d.io.read_triangle_mesh(path)
        pc = o3d.geometry.PointCloud(); pc.points = m.vertices
    else:
        pc = o3d.io.read_point_cloud(path)
    if len(pc.points) < 100:
        import trimesh
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.asarray(trimesh.load(path).vertices))
    return pc


def load_gear_cad():
    """Frozen gear CAD bundle: mesh, seeded target sample + normals, primitive features.

    Sampling uses open3d's sample_points_uniformly (uncertain determinism across
    builds) -> we therefore ALSO persist the sampled target + normals to
    code/minimal/artifacts/gear_target_{seed}.npz on first load and reuse the
    persisted copy afterwards, so every method sees byte-identical inputs.
    """
    art = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(art, exist_ok=True)
    npz = os.path.join(art, "gear_target_20260827.npz")
    mesh = o3d.io.read_triangle_mesh(f"{GEAR}/gear/model.obj")
    mesh.compute_vertex_normals()
    if os.path.exists(npz):
        z = np.load(npz)
        pts, nrm = z["pts"], z["nrm"]
    else:
        # NOTE (gpt-review fix): open3d 0.19 sample_points_uniformly exposes only
        # number_of_points / use_triangle_normal — NO seed kwarg. Determinism is
        # achieved by persisting the first draw to npz and reusing it everywhere.
        pc = mesh.sample_points_uniformly(number_of_points=N_TARGET)
        pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        pts, nrm = np.asarray(pc.points).copy(), np.asarray(pc.normals).copy()
        np.savez_compressed(npz, pts=pts, nrm=nrm)
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from features import load_features_from_json
    feats = load_features_from_json(f"{GEAR}/gear/extracted_features.json")
    return mesh, pts, nrm, feats


# ---------------------------------------------------------------- evaluation
def make_p2m_evaluator(mesh: o3d.geometry.PointCloud):
    """One-directional point-to-mesh distances (RaycastingScene), as used by all
    prior baseline scripts. Returns fn(pts)->distances(m)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    def p2m(pts: np.ndarray) -> np.ndarray:
        return scene.compute_distance(o3d.core.Tensor(np.asarray(pts, dtype=np.float32))).numpy().astype(np.float64)
    return p2m


def eval_pose(pts: np.ndarray, T: np.ndarray, p2m, inlier_mm: float = 2.0):
    """Standard pose metrics for real scans (no pose GT): p2m median/p95 + inlier %."""
    d = p2m(pts @ T[:3, :3].T + T[:3, 3])
    return {"p2m_median_mm": float(np.median(d)), "p2m_p95_mm": float(np.percentile(d, 95)),
            "inlier_pct": float(100.0 * np.mean(d < inlier_mm))}


# ---------------------------------------------------------------- shared FGR init
def fgr_coarse(src: o3d.geometry.PointCloud, tgt_pts: np.ndarray, tgt_nrm: np.ndarray):
    """The single shared FGR coarse start (voxel 1.0, frozen). Every refinement
    method — including GeoP2E — starts from THIS transform per scan."""
    voxel = CONFIGS["fgr_voxel"]
    sd = src.voxel_down_sample(voxel)
    sd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    fs = o3d.pipelines.registration.compute_fpfh_feature(sd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    tp = o3d.geometry.PointCloud(); tp.points = o3d.utility.Vector3dVector(tgt_pts); tp.normals = o3d.utility.Vector3dVector(tgt_nrm)
    td = tp.voxel_down_sample(voxel)
    ft = o3d.pipelines.registration.compute_fpfh_feature(td, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(
        maximum_correspondence_distance=CONFIGS["fgr_max_corr_dist"], use_absolute_scale=True)
    res = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(sd, td, fs, ft, opt)
    return res.transformation


# ---------------------------------------------------------------- gear symmetry
def gear_axis_and_symmetry(mesh: o3d.geometry.PointCloud, feats, tol_frac: float = 0.002):
    """Gear rotational-symmetry axis + discrete order, for symmetry-aware RRE.

    Axis: axis of the largest-radius CYLINDER primitive in the parsed CAD (the
    gear bore/hub axis). Order: DIRECT VERTEX-MATCH TEST with a TIGHT tolerance
    (gpt-review + pilot fix: tol=2% diagonal made every rotation 'match', i.e.
    the teeth were smaller than the tolerance, returning N=20 match=1.0; the
    gear is 17-tooth at tight tolerance). For each N in [2,60], rotate vertices
    by 2*pi/N about the axis, measure the fraction matched within
    tol_frac * model-diagonal; take the strict argmax N (fundamental order —
    non-group rotations drop sharply at tight tol). N_sym=1 if best match <0.6.
    Returns (unit axis, N_sym, match_fraction).
    """
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as Rot
    cyls = [f for f in feats if f.kind == "cylinder"]
    axis = max(cyls, key=lambda f: f.radius).axis if cyls else np.array([0.0, 0.0, 1.0])
    axis = axis / np.linalg.norm(axis)
    v = np.asarray(mesh.vertices)
    if len(v) > 20000:
        v = v[np.random.default_rng(0).choice(len(v), 20000, replace=False)]
    tree = cKDTree(v)
    tol = tol_frac * float(np.linalg.norm(np.ptp(v, axis=0)))
    best_N, best_match = 1, -1.0
    for N in range(2, 61):
        R = Rot.from_rotvec(axis * (2 * np.pi / N)).as_matrix()
        d, _ = tree.query(v @ R.T, k=1)
        m = float(np.mean(d < tol))
        if m > best_match:
            best_N, best_match = N, m
    if best_match < 0.6:
        return axis, 1, float(best_match)
    return axis, int(best_N), float(best_match)


def symmetry_rotations(axis: np.ndarray, N: int) -> list[np.ndarray]:
    """The N rotations about `axis` (identity included)."""
    from scipy.spatial.transform import Rotation as Rot
    if N <= 1:
        return [np.eye(3)]
    return [Rot.from_rotvec(axis * (2 * np.pi * k / N)).as_matrix() for k in range(N)]


def rre_rte_sym(T_pred: np.ndarray, T_gt: np.ndarray, sym_Rs: list[np.ndarray]):
    """Rotation/translation error minimised over the declared CAD-frame symmetry
    group. T_pred maps source->CAD, and a gear symmetry g acts in the CAD frame,
    so an equivalent solution is g @ T_pred: rotation R_g @ R_pred AND translation
    R_g @ t_pred (gpt-review fix: pre-multiply the FULL pose, not post-multiply
    the rotation with translation untouched).
    Returns (rre_sym, rte_sym, rre_raw, rte_raw)."""
    from solver import evaluate_alignment  # local import to avoid cycles
    rre_raw, rte_raw = evaluate_alignment(T_pred, T_gt)
    best_r, best_t = np.inf, np.inf
    for R in sym_Rs:
        Ts = np.eye(4)
        Ts[:3, :3] = R @ T_pred[:3, :3]
        Ts[:3, 3] = R @ T_pred[:3, 3]
        r, t = evaluate_alignment(Ts, T_gt)
        if r < best_r:
            best_r, best_t = r, t
    return best_r, best_t, rre_raw, rte_raw


# ---------------------------------------------------------------- misc
class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter(); return self
    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, default=float)
    print(f"saved {path}")
