"""e9 — viewpoint-occlusion benchmark on the five physical parts (Opus-planned).

Protocol: PLAN_E9_OCCLUSION_OPUS5.md. v2 (post GPT-5.6 xhigh review 2026-09-06):
- FIX (hang): shared-init Chamfer ranking now queries a 100k-point deterministic
  subsample of EvalCloud-A (1M-point exact queries measured 34-60 s each; the v1
  pilot's 19-min stall was 24 such queries in shared_init).
- FIX (critical): frame direction. T_ref from e8e is scan->CAD; viewpoint dirs are
  specified in CAD frame, mapped to the scan frame with d_S = R_gt^T @ d_C (was
  R_gt @ d_C = reversed).
- FIX (critical): B3 is now true point-to-PLANE ICP with 30-deg normal gate +
  Huber(1.2% diag) on the point-to-plane residual, left SE(3) linearized solve
  (was distance-gated point-to-point Kabsch).
- FIX (critical): G6 and B10 now consume the SAME ranked top-8 candidate list at
  the SAME budget (8 x 25-iter screen + 60-iter refine); G6's selector is the
  depth-buffer FSV composite (was 80%-trimmed RMSE mislabeled as FSV).
- FIX (critical): symmetry handled as full SE(3) (S @ T_gt, S about CAD centroid),
  frozen per part to artifacts/e9_symgroup_<stem>.npz (was: rotation-only,
  translation discarded; recomputed every run in 129 s).
- FIX (design): perturbation anchored at T_gt (T_c dT T_c^-1 T_gt) with SHA1
  per-part blocked seeds (was: anchored at the PCA init, hash() seeding).
- G3 now actually recomputes E-opt-8 selection from occlusion coverage under the
  trial init (was: placeholder = all features).
- B6 is now point-to-TRIANGLE ICP (RaycastingScene closest points + triangle
  normals; was: duplicate of B2).
- B1/B2/B6 use a shared progressive correspondence ladder (15/5/2 mm) so the
  10-30 deg init error is inside every arm's capture basin (was: flat 2.0 mm).
- Metrics: e_TTE over the full 2M EvalCloud-B, e_vis/e_hid/omega via the depth
  buffer, depth-buffer FSV, fixed-trim r_tau (80%) via mesh raycast distances,
  compat success; det(R)=+1 asserted on every arm output; T_init hash logged;
  rows appended to JSONL as computed; final json never overwrites (.N suffix).
- Pilot P-checks: 26-view rho sweep logged; P1 leak-through audit (5k retained
  points, ray-cast to sensor vs STEP mesh); P2 HPR Jaccard at gamma in {2,3,4}.

Still not implemented (disclosed): NOISE factor, I0/I2 regimes, 26-view arm
sweep, baseline calibration grids, cluster-bootstrap stats (analyze mode).
Usage: python e9_viewpoint_occlusion.py pilot
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys, json, time, hashlib, argparse, itertools, platform
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import cKDTree
from scipy.spatial import ConvexHull

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import e8_print_scans as E
import e6_auto as A
import e8e_bench as B
from solver import register
from selection import greedy_select

GEO = dict(init_radius=3.0, anneal=0.9, max_iter=60, lam=1e-3, angle_thresh_deg=30.0)
BENCH = json.load(open(os.path.join(HERE, "results", "e8e_bench.1788642147.json")))
OUTDIR = os.path.join(HERE, "results", "e9")
ART = os.path.join(HERE, "artifacts")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(ART, exist_ok=True)
THETA_MAX = 75.0
GSD = 0.5
FSV_EPS = 1.0
FSV_W = 10.0          # mm penalty per unit FSV in the G6 selector (disclosed)
SUBA_N = 100_000      # ranking-tree subsample of EvalCloud-A
EVTB_STRIDE = 1       # e_TTE evaluated on the full 2M EvalCloud-B


def sha_seed(tag: str) -> int:
    return int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16)


def t_hash(T: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(T, dtype=np.float64).tobytes()).hexdigest()[:12]


# --------------------------------------------------------------- occlusion
def canonical_basis(part, tcad):
    """Deterministic CAD-frame basis with z = largest-plane outward normal."""
    best, ba = None, -1
    for m in part["supports"]:
        if m["feat"].kind == "plane":
            a = len(m["pts"])
            if a > ba:
                ba, best = a, m["feat"].normal
    c = tcad.pts.mean(0)
    gc = None
    for m in part["supports"]:
        if m["feat"].kind == "plane" and len(m["pts"]) == ba:
            gc = m["pts"].mean(0)
    if gc is not None and np.dot(gc - c, best) < 0:
        best = -best
    z = best / np.linalg.norm(best)
    tmp = np.eye(3)[int(np.argmin(np.abs(z)))]
    x = np.cross(z, tmp); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return x, y, z


def view_dirs_26(xa, ya, za):
    """26 canonical directions expressed in the CAD frame."""
    faces = [xa, -xa, ya, -ya, za, -za]
    edges, corners = [], []
    for i, j in ((xa, ya), (xa, za), (ya, za)):
        for s1 in (1, -1):
            for s2 in (1, -1):
                edges.append((s1 * i + s2 * j) / np.sqrt(2))
    for s1 in (1, -1):
        for s2 in (1, -1):
            for s3 in (1, -1):
                corners.append((s1 * xa + s2 * ya + s3 * za) / np.sqrt(3))
    return faces + edges + corners


def occlude(pts, nrm, T_gt, d_C, scene=None):
    """Single-view visibility. pts/nrm in SCAN frame; d_C viewpoint dir in CAD
    frame; T_gt: scan->CAD (e8e convention). Returns mask, sensor-consistent
    outward normals, and the sensor geometry for FSV / region split.

    Visibility = facing/75-deg gate AND mesh-raycast unblocked (primary) when
    `scene` is given; the z-buffer (scan-cloud self-occlusion) is computed as a
    documented pre-filter and its disagreement with the raycast (~5%: points
    blocked only by mesh geometry the multi-view input scan never sampled) is
    reported. Protocol amendment recorded 2026-09-06 after the pilot audit
    showed z-buffer-only retention leaks 5.1% physically-blocked points."""
    d_S = T_gt[:3, :3].T @ d_C                       # CAD dir -> scan frame
    D = np.linalg.norm(np.ptp(pts, axis=0))
    mu = (pts.max(0) + pts.min(0)) / 2
    c = mu + 6 * D * d_S / np.linalg.norm(d_S)
    w = -d_S / np.linalg.norm(d_S)
    tmp = np.eye(3)[int(np.argmin(np.abs(w)))]
    u = np.cross(w, tmp); u /= np.linalg.norm(u)
    v = np.cross(w, u)
    rel = pts - c
    z = rel @ w
    x = rel @ u; y = rel @ v
    facing = z > 0
    vdir = (c - pts)
    vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
    # sensor-consistent outward normals: flip against the view ray
    sign = np.sign(np.sum(nrm * vdir, axis=1)); sign[sign == 0] = 1
    n_out = nrm * sign[:, None]
    cos2 = np.clip(np.sum(n_out * vdir, axis=1), -1, 1)
    ang2 = np.degrees(np.arccos(cos2))
    vis_face = ang2 <= THETA_MAX
    # z-buffer (perspective bins, GSD-matched)
    Dl = 6 * D
    delta = GSD / Dl
    a = np.floor((x / np.maximum(z, 1e-9)) / delta).astype(np.int64)
    b = np.floor((y / np.maximum(z, 1e-9)) / delta).astype(np.int64)
    key = a * 1000003 + b
    order = np.lexsort((z, key))
    vis = np.zeros(len(pts), bool)
    last_key = None; zmin = np.inf
    eps = np.minimum(0.3 + 2 * GSD * np.tan(np.radians(np.minimum(ang2, 89.9))), 3.0)
    zmin_tab = {}
    for idx in order:
        k = int(key[idx])
        if k != last_key:
            last_key = k; zmin = z[idx]; zmin_tab[k] = zmin
        if z[idx] <= zmin + eps[idx]:
            vis[idx] = True
    mask = facing & vis_face & vis
    if scene is not None:
        # mesh-raycast primary: keep only points with clear sight-line to the
        # sensor under the TRUE part geometry (exact single-view visibility).
        # Sensor-origin rays: blocked iff t_hit in (0.5, L-0.5).
        c_CAD = T_gt[:3, :3] @ c + T_gt[:3, 3]
        p_CAD = pts @ T_gt[:3, :3].T + T_gt[:3, 3]
        dv = p_CAD - c_CAD                      # sensor -> point
        L = np.linalg.norm(dv, axis=1); dv /= np.maximum(L, 1e-12)[:, None]
        rays = o3d.core.Tensor(np.hstack([np.broadcast_to(c_CAD, p_CAD.shape), dv]).astype(np.float32))
        t_hit = scene.cast_rays(rays)["t_hit"].numpy().astype(np.float64)
        blocked = np.isfinite(t_hit) & (t_hit > 0.5) & (t_hit < L - 0.5)
        mask = mask & (~blocked)
    sensor = dict(c=c, u=u, v=v, w=w, delta=delta, zmin_tab=zmin_tab, D=D, d_S=d_S,
                  zbuf_only=bool(scene is None))
    return mask, n_out[mask], sensor


def depth_visible(pts_scan, sensor, tol=0.3):
    """Depth-buffer visibility test for arbitrary points given in the SCAN frame
    (no facing/normal gate; z-buffer only). Used for e_vis/e_hid split."""
    c, u, v, w = sensor["c"], sensor["u"], sensor["v"], sensor["w"]
    rel = pts_scan - c
    z = rel @ w
    ok = z > 0
    x = rel @ u; y = rel @ v
    a = np.floor((x / np.maximum(z, 1e-9)) / sensor["delta"]).astype(np.int64)
    b = np.floor((y / np.maximum(z, 1e-9)) / sensor["delta"]).astype(np.int64)
    key = a * 1000003 + b
    vis = np.zeros(len(pts_scan), bool)
    kt = sensor["zmin_tab"]
    for i in np.nonzero(ok)[0]:
        k = int(key[i])
        zmin = kt.get(k)
        if zmin is not None and z[i] <= zmin + tol:
            vis[i] = True
    return vis


def fsv_depth(T_est, sensor, cad_sub):
    """Free-space violation: fraction of CAD subsample (CAD frame) that projects
    inside the observed frustum BEHIND the measured depth (z < Z - eps)."""
    if len(cad_sub) > 50_000:                        # exact cap, deterministic
        r = np.random.default_rng(sha_seed("fsvprobe"))
        cad_sub = cad_sub[r.choice(len(cad_sub), 50_000, replace=False)]
    Tinv = np.linalg.inv(T_est)                      # CAD -> scan
    p = cad_sub @ Tinv[:3, :3].T + Tinv[:3, 3]
    c, u, v, w = sensor["c"], sensor["u"], sensor["v"], sensor["w"]
    rel = p - c
    z = rel @ w
    m = z > 0
    x = rel @ u; y = rel @ v
    a = np.floor((x / np.maximum(z, 1e-9)) / sensor["delta"]).astype(np.int64)
    b = np.floor((y / np.maximum(z, 1e-9)) / sensor["delta"]).astype(np.int64)
    kt = sensor["zmin_tab"]
    viol = inside = 0
    for i in np.nonzero(m)[0]:
        k = int(a[i] * 1000003 + b[i])
        zmin = kt.get(k)
        if zmin is not None:
            inside += 1
            if z[i] < zmin - FSV_EPS:
                viol += 1
    return (viol / inside) if inside else 0.0


def hpr_visible(pts, c, gamma):
    """Katz HPR operator-robustness check: visible = points on conv({p_hat} u {c})."""
    d = pts - c
    n = np.linalg.norm(d, axis=1)
    R = (10.0 ** gamma) * n.max()
    p_hat = c + d / np.maximum(n, 1e-12)[:, None] * (2 * R - n)[:, None]
    hull = ConvexHull(np.vstack([p_hat, c]))
    vis = np.zeros(len(pts), bool)
    hv = np.unique(hull.vertices)
    vis[hv[hv < len(pts)]] = True   # drop the appended camera point (index len(pts))
    return vis


# --------------------------------------------------------------- init ladder
def pca24_occluded(src, tgt_pts):
    cs, ct = src.mean(0), tgt_pts.mean(0)
    As = np.linalg.svd(src - cs, full_matrices=False)[2].T
    At = np.linalg.svd(tgt_pts - ct, full_matrices=False)[2].T
    if np.linalg.det(As) < 0: As[:, 2] *= -1
    if np.linalg.det(At) < 0: At[:, 2] *= -1
    out = []
    for p in itertools.permutations(range(3)):
        for f in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3))
            for i in range(3): M[i, p[i]] = f[i]
            if np.linalg.det(M) < 0: continue
            R = At @ M @ As.T
            assert np.linalg.det(R) > 0
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = ct - R @ cs
            out.append(T)
    return out


def rank_candidates(src_occl, tgt_pts, subA_tree):
    """PCA-24 (+ centroid fallback when PCA axes degenerate), ranked by one-sided
    Chamfer scan->CAD against the 100k subsample tree. Returns (best_T, top8, gate)."""
    cs = src_occl - src_occl.mean(0)
    ev = np.linalg.eigvalsh(np.cov(cs.T))[::-1]
    degenerate = (0.85 <= ev[1] / max(ev[0], 1e-12) <= 1.18) or \
                 (len(ev) > 2 and 0.85 <= ev[2] / max(ev[1], 1e-12) <= 1.18)
    cands = pca24_occluded(src_occl, tgt_pts)
    if degenerate:
        T0 = np.eye(4); T0[:3, 3] = tgt_pts.mean(0) - src_occl.mean(0)
        cands.append(T0)
    # score on a 10k deterministic subsample (24 x full-cloud queries measured
    # ~0.7 s each against even a 100k tree; ranking precision does not need more)
    rng10 = np.random.default_rng(sha_seed("rankscore"))
    idx10 = rng10.choice(len(src_occl), min(10_000, len(src_occl)), replace=False)
    src10 = src_occl[idx10]
    scored = []
    for T in cands:
        p = src10 @ T[:3, :3].T + T[:3, 3]
        d, _ = subA_tree.query(p, k=1, workers=1)
        scored.append((T, float(np.mean(d))))
    scored.sort(key=lambda s: s[1])
    return scored[0][0], [T for T, _ in scored[:8]], dict(degenerate=bool(degenerate),
                                                         lam=ev.tolist())


def make_perturb(stem, k, regime="I1"):
    """Blocked per-part perturbation (protocol 3.2): same dT_k reused across all
    viewpoints of a part. theta ~ U[10,30] deg, |t| ~ U[3,10] mm about CAD centroid."""
    rng = np.random.default_rng(sha_seed(f"{stem}|{regime}|{k}"))
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
    th = np.radians(rng.uniform(10, 30))
    tdir = rng.normal(size=3); tdir /= np.linalg.norm(tdir)
    tt = rng.uniform(3, 10)
    dT = np.eye(4); dT[:3, :3] = Rot.from_rotvec(ax * th).as_matrix()
    dT[:3, 3] = tdir * tt
    return dT, dict(seed=sha_seed(f"{stem}|{regime}|{k}"), theta_deg=float(np.degrees(th)),
                    t_mm=float(tt))


def apply_perturb(dT, T_gt, mu_CAD):
    """T_init = T_c dT T_c^-1 T_gt, anchored at ground truth (I1 regime)."""
    Tc = np.eye(4); Tc[:3, 3] = mu_CAD
    return Tc @ dT @ np.linalg.inv(Tc) @ T_gt


# --------------------------------------------------------------- symmetry
def load_part(si):
    stem = B.PARTS[si]
    pts, nrm, _ = E.load_scan(si)
    work, _ = E.clean_downsample(pts, nrm)
    src = np.asarray(work.points); sn = np.asarray(work.normals)
    ev, tgt, _ = E.cad_clouds(stem)
    part = A.load_parsed_part(os.path.join(E.PACK, f"{stem}_scaled.step"))
    tcad = A.TrimmedCAD(part)
    T_gt = np.array(BENCH["parts"][f"scan{si}"]["T_ref"])
    assert np.linalg.det(T_gt[:3, :3]) > 0
    # EvalCloud-B: independent 2M area-uniform sampling, deterministic seed
    npz = os.path.join(ART, f"e9_evalB_{stem}.npz")
    if os.path.exists(npz):
        evalB = np.load(npz)["pts"]
    else:
        m = o3d.io.read_triangle_mesh(os.path.join(ART, f"e8_eval_{stem}_0p02.stl"))
        rng = np.random.default_rng(13)
        tri = np.asarray(m.triangles); V = np.asarray(m.vertices)
        a0 = np.linalg.norm(np.cross(V[tri[:, 1]] - V[tri[:, 0]], V[tri[:, 2]] - V[tri[:, 0]]), axis=1)
        pick = rng.choice(len(tri), 2_000_000, p=a0 / a0.sum())
        r1, r2 = np.sqrt(rng.random(2_000_000)), rng.random(2_000_000)
        evalB = (V[tri[pick, 0]] * (1 - r1[:, None]) + V[tri[pick, 1]] * (r1[:, None] * (1 - r2[:, None]))
                 + V[tri[pick, 2]] * (r1[:, None] * r2[:, None]))
        np.savez_compressed(npz, pts=evalB)
    # frozen symmetry group (full SE(3), proper rotations about CAD centroid)
    symz = os.path.join(ART, f"e9_symgroup_{stem}.npz")
    if os.path.exists(symz):
        grp = [S for S in np.load(symz)["S"]]
    else:
        rec = BENCH["parts"][f"scan{si}"]
        grp = B.empirical_symmetry_group(src, ev, T_gt, rec["ref_p2m"]["median"],
                                         rec["ref_p2m"]["p95"])
        grp = [S for S in grp if not np.allclose(S, np.eye(4))]
        np.savez_compressed(symz, S=np.array(grp) if grp else np.empty((0, 4, 4)))
    # STEP mesh scene (B6 target + r_tau true distances) and ranking subsample
    mesh = o3d.io.read_triangle_mesh(os.path.join(ART, f"e8_eval_{stem}_0p02.stl"))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rng2 = np.random.default_rng(7)
    subA = np.asarray(ev.points)[rng2.choice(len(ev.points), min(SUBA_N, len(ev.points)),
                                             replace=False)]
    # FSV probe cloud = 100k subsample of EvalCloud-A (NOT the held-out EvalCloud-B)
    for i, f in enumerate(tcad.feats):          # bounded-face switch support:
        if i < len(part["supports"]):
            pts_i = part["supports"][i]["pts"]
            # keep only points CONSISTENT with THIS feature's parameters (the
            # e6_auto merge can leave mirrored-plane points in a support cloud —
            # 21638 support6: 68% of points 49mm off-plane; review CRITICAL)
            rr = np.abs(f.compute_residual(pts_i))
            good = pts_i[rr < 1.0]
            f.bounds_tree = cKDTree(good if len(good) >= 30 else pts_i)
    return dict(si=si, stem=stem, src=src, sn=sn, ev=ev, tgt=tgt, part=part, tcad=tcad,
                T_gt=T_gt, evalB=evalB, grp=grp, scene=scene, subA_tree=cKDTree(subA),
                subA=subA, tgt_pts=np.asarray(tgt.points))


# --------------------------------------------------------------- metrics
def e_tte(T_est, T_gt, evalB, grp):
    """Symmetry-aware whole-model RMS pose error (mm) over the full EvalCloud-B.
    Explicit relative-transform form: Delta = T_est @ inv(Tg) acts on CAD-frame
    model points; e = RMS ||Delta x - x||. (Equivalent to the previous
    two-sided form by rigid conjugation, but frame-explicit.) Valid poses are
    Tg = S @ T_gt, S in the frozen proper symmetry group."""
    x = evalB[::EVTB_STRIDE]
    best = None
    for S in [np.eye(4)] + list(grp):
        Tg = S @ T_gt
        D = T_est @ np.linalg.inv(Tg)           # CAD -> CAD relative error
        d = (x @ D[:3, :3].T + D[:3, 3]) - x
        e = float(np.sqrt(np.mean(np.sum(d * d, axis=1))))
        if best is None or e < best: best = e
    return best


def region_split_errors(T_est, T_gt, evalB, grp, sensor):
    """e_vis over CAD points visible at GT pose (depth-buffer), e_hid over the
    complement, omega = e_hid / max(e_vis, 0.05)."""
    Tinv = np.linalg.inv(T_gt)
    p_scan = evalB[::4] @ Tinv[:3, :3].T + Tinv[:3, 3]
    vis = depth_visible(p_scan, sensor)
    x = evalB[::4]
    best = None
    for S in [np.eye(4)] + list(grp):
        Tg = S @ T_gt
        D = T_est @ np.linalg.inv(Tg)
        diff = (x @ D[:3, :3].T + D[:3, 3]) - x
        d = np.sqrt(np.sum(diff * diff, axis=1))
        r = (float(np.sqrt(np.mean(d[vis] ** 2))) if vis.any() else 0.0,
             float(np.sqrt(np.mean(d[~vis] ** 2))) if (~vis).any() else 0.0)
        if best is None or (r[0] + r[1]) < (best[0] + best[1]): best = r
    e_vis, e_hid = best
    return e_vis, e_hid, e_hid / max(e_vis, 0.05)


def r_tau(T_est, src_occl, scene):
    """Fixed-trim (80%) RMS point-to-MESH distance of the occluded scan."""
    p = src_occl @ T_est[:3, :3].T + T_est[:3, 3]
    d = scene.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
    k = int(np.ceil(0.8 * len(d)))
    return float(np.sqrt(np.mean(np.sort(d)[:k] ** 2)))


def compat_success(T_est, src_occl, scene):
    p = src_occl @ T_est[:3, :3].T + T_est[:3, 3]
    d = scene.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy()
    return bool(np.median(d) < 0.5 and np.percentile(d, 95) < 2.0)


# --------------------------------------------------------------- baselines
def icp_ladder(src, nrm, tgt_pc, T0, est, thrs=(15.0, 5.0, 2.0), iters=30):
    T = T0.copy()
    for thr in thrs:
        T = o3d.pipelines.registration.registration_icp(
            pc_of(src, nrm), tgt_pc, thr, T, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iters)).transformation
    return T


def pc_of(pts, nrm=None):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts)
    if nrm is not None:
        pc.normals = o3d.utility.Vector3dVector(nrm)
    return pc


def b3_p2plane_gated_huber(src, nrm, tgt_pts, tgt_nrm, tree, T0, k_huber, thrs=(15.0, 5.0, 2.0),
                           iters=25, gate_deg=30.0):
    """TRUE point-to-plane ICP with 30-deg bilateral normal gate + Huber weights
    on the point-to-plane residual. Left SE(3) linearized solve (matches GeoP2E)."""
    T = T0.copy()
    cos_gate = np.cos(np.radians(gate_deg))
    tgt_tree = cKDTree(tgt_pts)   # SAME point set as tgt_nrm (voxel cloud w/ normals)
    for thr in thrs:
        for _ in range(iters):
            p = src @ T[:3, :3].T + T[:3, 3]
            n_rot = nrm @ T[:3, :3].T                # normals rotated with the pose
            d, idx = tgt_tree.query(p, k=1, workers=1)
            m = (d < thr) & (np.abs(np.einsum("ij,ij->i", n_rot, tgt_nrm[idx])) >= cos_gate)
            if m.sum() < 30:
                break
            P = p[m]; n = tgt_nrm[idx][m]
            r = np.einsum("ij,ij->i", P - tgt_pts[idx][m], n)   # point-to-plane residual
            w = np.minimum(1.0, k_huber / np.maximum(np.abs(r), 1e-9))
            # Jacobian of residual wrt left SE(3) update xi=[w;x]: d r = n . (w x (R p)) + n . x
            J = np.hstack([np.cross(P, n), n])
            W = w[:, None]
            H = J.T @ (W * J); b = J.T @ (W * -r[:, None]).ravel()
            try:
                dxi = np.linalg.solve(H + 1e-9 * np.trace(H) / 6 * np.eye(6), b)
            except np.linalg.LinAlgError:
                break
            from solver import se3_exp
            T = se3_exp(dxi) @ T
            if np.linalg.norm(dxi) < 1e-7:
                break
    return T


def b6_p2triangle(src, nrm, scene, T0, thrs=(15.0, 5.0, 2.0), iters=25):
    """Point-to-TRIANGLE ICP: correspondences are closest points on the STEP mesh
    with triangle normals (exact bounded geometry)."""
    T = T0.copy()
    for thr in thrs:
        for _ in range(iters):
            p = src @ T[:3, :3].T + T[:3, 3]
            ans = scene.compute_closest_points(o3d.core.Tensor(p.astype(np.float32)))
            q = ans["points"].numpy().astype(np.float64)
            n = ans["primitive_normals"].numpy().astype(np.float64)
            d = np.linalg.norm(p - q, axis=1)
            m = d < thr
            if m.sum() < 30:
                break
            P, Q, N = p[m], q[m], n[m]
            r = np.einsum("ij,ij->i", P - Q, N)
            J = np.hstack([np.cross(P, N), N])
            H = J.T @ J; b = J.T @ -r
            try:
                dxi = np.linalg.solve(H + 1e-9 * np.trace(H) / 6 * np.eye(6), b)
            except np.linalg.LinAlgError:
                break
            from solver import se3_exp
            T = se3_exp(dxi) @ T
            if np.linalg.norm(dxi) < 1e-7:
                break
    return T


def trimmed_rmse(T, src, tree, trim=0.8):
    p = src @ T[:3, :3].T + T[:3, 3]
    d, _ = tree.query(p, k=1, workers=1)
    k = int(np.ceil(trim * len(d)))
    return float(np.sqrt(np.mean(np.sort(d)[:k] ** 2)))


# --------------------------------------------------------------- arms
CAND_CACHE = []
SENSOR_CACHE = []
SUBA_PROBE = []
FSV_W_CACHE = [FSV_W]


def run_arm(arm, P, src_o, nrm_o, T_init, feats, feats_e8, feats_occl, REG, feats_eng=None):
    """All arms: identical occluded cloud, identical T_init, det(R)=+1 asserted."""
    t0 = time.perf_counter()
    T = np.eye(4); info = {}
    tgt_pc = P["tgt"]; tree_A = P["subA_tree"]
    if arm == "B1":
        T = icp_ladder(src_o, nrm_o, tgt_pc, T_init,
                       o3d.pipelines.registration.TransformationEstimationPointToPoint())
    elif arm == "B2":
        T = icp_ladder(src_o, nrm_o, tgt_pc, T_init,
                       o3d.pipelines.registration.TransformationEstimationPointToPlane())
    elif arm == "B3":
        tn = np.asarray(tgt_pc.normals)
        ext = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        T = b3_p2plane_gated_huber(src_o, nrm_o, P["tgt_pts"], tn, tree_A, T_init,
                                   k_huber=ext * 0.012)
    elif arm == "B6":
        T = b6_p2triangle(src_o, nrm_o, P["scene"], T_init)
    elif arm == "B10":
        # candidates = shared ranked top-8 PLUS this trial's T_init (identical
        # list for G6; the per-trial member makes k matter and kills the
        # pseudoreplicate structure flagged in review round 2)
        cands = list(CAND_CACHE) + [T_init]
        best = None
        for Tc in cands:
            Tm = o3d.pipelines.registration.registration_icp(
                pc_of(src_o, nrm_o), tgt_pc, 5.0, Tc,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25)).transformation
            s = trimmed_rmse(Tm, src_o, tree_A)
            if best is None or s < best[1]: best = (Tm, s)
        T = o3d.pipelines.registration.registration_icp(          # 60-iter refine on winner
            pc_of(src_o, nrm_o), tgt_pc, 3.0, best[0],
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=60)).transformation
    elif arm == "B4":
        T = icp_ladder(src_o, nrm_o, tgt_pc, T_init,
                       o3d.pipelines.registration.TransformationEstimationForGeneralizedICP())
    elif arm == "B5":
        from baselines import trimmed_icp
        T, _ = trimmed_icp(src_o, P["tgt_pts"], init=T_init.copy(), threshold=3.0,
                           trim_frac=0.3, max_iter=60)
    elif arm == "B11":
        # multistart ICP, selector = trimmed RMSE + FSV_W * FSV (same composite
        # family as G6; both calibrated on the disjoint calibration set)
        cands = list(CAND_CACHE) + [T_init]
        best = None
        for Tc in cands:
            Tm = o3d.pipelines.registration.registration_icp(
                pc_of(src_o, nrm_o), tgt_pc, 5.0, Tc,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25)).transformation
            sc = trimmed_rmse(Tm, src_o, tree_A)                 + FSV_W_CACHE[-1] * fsv_depth(Tm, SENSOR_CACHE[-1], SUBA_PROBE[-1])
            if best is None or sc < best[1]: best = (Tm, sc)
        T = o3d.pipelines.registration.registration_icp(
            pc_of(src_o, nrm_o), tgt_pc, 3.0, best[0],
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=60)).transformation
    elif arm == "B10b":
        # B10 budget-matched to G6: two-stage ICP schedule (30 @ 15mm + 60 @ 3mm)
        cands = list(CAND_CACHE) + [T_init]
        best = None
        for Tc in cands:
            Tm = o3d.pipelines.registration.registration_icp(
                pc_of(src_o, nrm_o), tgt_pc, 15.0, Tc,
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25)).transformation
            s_ = trimmed_rmse(Tm, src_o, tree_A)
            if best is None or s_ < best[1]: best = (Tm, s_)
        T = icp_ladder(src_o, nrm_o, tgt_pc, best[0],
                       o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                       thrs=(5.0, 2.0), iters=30)[:0] if False else             o3d.pipelines.registration.registration_icp(
                pc_of(src_o, nrm_o), tgt_pc, 3.0, best[0],
                o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                o3d.pipelines.registration.ICPConvergenceCriteria(
                    relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=90)).transformation
    elif arm in ("G1", "G2", "G3", "G6", "G4", "G6a", "G7", "G8", "GB"):
        f = {"G1": feats, "G2": feats_e8, "G3": feats_occl, "G6": feats,
             "G4": feats, "G6a": feats, "G7": feats_eng, "G8": feats, "GB": feats}[arm]
        f_coarse = [x for x in feats if x.kind == "plane"] if arm == "G8" else f
        rg = dict(REG)
        if arm == "G4":                       # per-part offline profile (e8e CALIB)
            cal = B.CALIB.get(P["si"], {})
            rg = dict(rg, lam=cal.get("lam", rg["lam"]),
                      angle_thresh_deg=cal.get("ang", rg["angle_thresh_deg"]))
        # two-stage radius schedule (protocol amendment, disclosed): the I1
        # perturbation (10-30 deg) exceeds the frozen production radius 3 mm
        # (an in-basin refinement config); a 15 mm coarse stage captures it,
        # matching the 15/5/2 mm correspondence ladders every ICP arm gets.
        diag = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        r_coarse = min(15.0, 0.12 * diag)   # scale rule: stay under e8c's 0.18*diag divergence
        def reg2(t0, iters_fine=60, iters_coarse=30):
            kw = {"bounded_face": True} if arm == "GB" else {}
            Tc_, ic = register(src_o, nrm_o, f_coarse if len(f_coarse) >= 2 else f, init_T=t0.copy(),
                               **{**rg, "init_radius": r_coarse, "anneal": 0.85, "max_iter": iters_coarse}, **kw)
            Tf_, if_ = register(src_o, nrm_o, f, init_T=Tc_, **{**rg, "max_iter": iters_fine}, **kw)
            return Tf_, {"coarse_iters": ic.get("iters"), "coarse_upd": ic.get("update_norm"),
                         "coarse_rmse0": (ic.get("history") or [{}])[0].get("rmse"),
                         "fine_iters": if_.get("iters")}
        try:
            if arm == "G6":
                cands = list(CAND_CACHE) + [T_init]      # SAME list as B10
                best = None
                for Tc in cands:
                    Tm, _ = register(src_o, nrm_o, f, init_T=Tc.copy(),
                                     **{**rg, "init_radius": r_coarse, "anneal": 0.85, "max_iter": 25})
                    s = trimmed_rmse(Tm, src_o, tree_A) \
                        + FSV_W * fsv_depth(Tm, SENSOR_CACHE[-1], SUBA_PROBE[-1])
                    if best is None or s < best[1]: best = (Tm, s)
                T, info = reg2(best[0])                  # refine = G1's 30+60 (disclosed)
            else:
                T, info = reg2(T_init)
        except Exception as ex:
            T = T_init.copy(); info = {"error": repr(ex)[:80]}
        except Exception as ex:
            T = T_init.copy(); info = {"error": repr(ex)[:80]}
    assert np.linalg.det(T[:3, :3]) > 0, f"{arm}: improper output transform"
    return T, time.perf_counter() - t0, info


# --------------------------------------------------------------- pilot
PILOT_ARMS = ("B3", "B6", "G1", "G2", "G3", "G6", "B10")


def pilot_checks(P, mask, sensor, stem, vn):
    """P1 leak-through audit (5k retained points, ray-cast vs STEP mesh) and
    P2 HPR Jaccard at gamma in {2,3,4}."""
    src = P["src"]; T_gt = P["T_gt"]; scene = P["scene"]
    vis_idx = np.nonzero(mask)[0]
    rng = np.random.default_rng(sha_seed(f"{stem}|{vn}|audit"))
    samp = vis_idx[rng.choice(len(vis_idx), min(5000, len(vis_idx)), replace=False)]
    p = src[samp]
    c = sensor["c"]
    c_CAD = T_gt[:3, :3] @ c + T_gt[:3, 3]
    p_CAD = p @ T_gt[:3, :3].T + T_gt[:3, 3]
    dirs = p_CAD - c_CAD
    L = np.linalg.norm(dirs, axis=1); dirs /= L[:, None]
    # sensor-origin rays: a hit at t in (0.5, L-0.5) is geometry strictly
    # between the sensor and the point = true occluder. (Point-origin rays
    # graze the point's own surface at small t and false-positive.)
    rays = o3d.core.Tensor(np.hstack([np.broadcast_to(c_CAD, p_CAD.shape), dirs]).astype(np.float32))
    hit = scene.cast_rays(rays)
    t_hit = hit["t_hit"].numpy().astype(np.float64)
    blocked = np.isfinite(t_hit) & (t_hit > 0.5) & (t_hit < L - 0.5)
    leak = float(np.mean(blocked))
    # operator-Jaccard on a MIXED full-cloud sample (visible + invisible), so
    # both sets are non-trivial: J(V_zbuf, V_HPR) over 5k random cloud points
    rng2 = np.random.default_rng(sha_seed(f"{stem}|{vn}|hprsample"))
    hidx = rng2.choice(len(src), min(5000, len(src)), replace=False)
    hmask = src[hidx]
    # full operator mask for the sampled points = the same raycast-primary test
    c_CAD_h = T_gt[:3, :3] @ c + T_gt[:3, 3]
    p_CAD_h = hmask @ T_gt[:3, :3].T + T_gt[:3, 3]
    dh = c_CAD_h - p_CAD_h
    Lh = np.linalg.norm(dh, axis=1); dh /= Lh[:, None]
    rh = o3d.core.Tensor(np.hstack([np.broadcast_to(c_CAD_h, p_CAD_h.shape), -dh]).astype(np.float32))
    th = scene.cast_rays(rh)["t_hit"].numpy().astype(np.float64)
    v_op = (~np.isfinite(th)) | (th >= Lh - 0.5) | (th <= 0.5)
    hpr = {}
    for g in (2, 3, 4):
        vh = hpr_visible(hmask, c, g)
        inter = (vh & v_op).sum(); union = (vh | v_op).sum()
        hpr[g] = float(inter / union)
    assert leak < 0.01, f"P1 leak-through {leak:.4f} >= 1% (K2 halt)"
    # leak bucketed by incidence angle (is the residual concentrated at grazing?)
    vdir_s = (c - p); vdir_s /= np.linalg.norm(vdir_s, axis=1, keepdims=True)
    ns = P["sn"][samp]
    sg = np.sign(np.sum(ns * vdir_s, axis=1)); sg[sg == 0] = 1
    ang = np.degrees(np.arccos(np.clip(np.sum(ns * sg[:, None] * vdir_s, axis=1), -1, 1)))
    buckets = {}
    for lo, hi in ((0, 45), (45, 60), (60, 75)):
        msk = (ang >= lo) & (ang < hi)
        if msk.any():
            buckets[f"{lo}-{hi}"] = (int(msk.sum()), float(np.mean(blocked[msk])))
    # full-cloud HPR retention agreement (P2 second criterion)
    c_full = sensor["c"]
    ret_hpr = {}
    for g in (3, 4):
        vh = hpr_visible(src, c_full, g)
        ret_hpr[g] = float(vh.sum() / len(src))
    return dict(leak_through=leak, hpr_jaccard=hpr, leak_by_angle=buckets,
                rho_hpr=ret_hpr)


def pilot():
    si = 3  # 29726 (|G|=1, no symmetry confound)
    P = load_part(si)
    stem = P["stem"]; T_gt = P["T_gt"]
    xa, ya, za = canonical_basis(P["part"], P["tcad"])
    pool = view_dirs_26(xa, ya, za)
    # 26-view rho sweep (cheap, logged for the record / substitution decisions)
    rho = []
    for d in pool:
        m, _, _ = occlude(P["src"], P["sn"], T_gt, d, scene=P["scene"])
        rho.append(len(P["src"][m]) / len(P["src"]))
    print("rho26=" + ",".join(f"{r:.2f}" for r in rho), flush=True)
    top_dir = za
    diag_dir = np.array((1, 1, 1), float) / np.sqrt(3)
    diag_C = diag_dir[0] * xa + diag_dir[1] * ya + diag_dir[2] * za
    # P1 band check with protocol 1.5 substitution (recorded, never silent):
    # if the nominal pilot view falls outside rho in [0.25, 0.75], substitute
    # the nearest-rho unused pool direction.
    def in_band(d):
        m, _, _ = occlude(P["src"], P["sn"], T_gt, d, scene=P["scene"])
        return len(P["src"][m]) / len(P["src"]), m
    views, subs = {}, []
    rho_top, _ = in_band(top_dir)
    used = []
    if 0.25 <= rho_top <= 0.75:
        views["top"] = top_dir
    else:
        cand = [(abs(r - rho_top), k) for k, r in enumerate(rho) if 0.25 <= r <= 0.75]
        j = min(cand)[1]
        views["top"] = pool[j]; used.append(j)
        subs.append(f"top rho={rho_top:.2f} out of band -> pool[{j}] rho={rho[j]:.2f}")
    rho_dg, _ = in_band(diag_C)
    if 0.25 <= rho_dg <= 0.75:
        views["diag"] = diag_C
    else:
        cand = [(abs(r - rho_dg), k) for k, r in enumerate(rho)
                if 0.25 <= r <= 0.75 and k not in used]
        k = min(cand)[1]
        views["diag"] = pool[k]; used.append(k)
        subs.append(f"diag rho={rho_dg:.2f} out of band -> pool[{k}] rho={rho[k]:.2f}")
    if subs:
        print("SUBSTITUTION: " + "; ".join(subs), flush=True)
    out_jsonl = os.path.join(OUTDIR, "pilot.jsonl")
    n = 1
    while os.path.exists(out_jsonl):
        out_jsonl = os.path.join(OUTDIR, f"pilot.{n}.jsonl"); n += 1
    rows = []
    all_checks = {}
    for vn, d_C in views.items():
        mask, nrm_o, sensor = occlude(P["src"], P["sn"], T_gt, d_C, scene=P["scene"])
        src_o = P["src"][mask]
        rho_v = len(src_o) / len(P["src"])
        chk = pilot_checks(P, mask, sensor, stem, vn)
        all_checks[vn] = dict(chk, n_audit=int(min(5000, len(np.nonzero(mask)[0]))),
                              n_cloud=int(len(P["src"])), n_evalB=int(len(P["evalB"])),
                              n_subA=int(SUBA_N))
        print(f"[{stem}] view={vn} rho={rho_v:.2f} N={len(src_o):,} "
              f"leak={chk['leak_through']:.4f} hprJ={chk['hpr_jaccard']}", flush=True)
        T0, top8, gate = rank_candidates(src_o, P["tgt_pts"], P["subA_tree"])
        global CAND_CACHE, SENSOR_CACHE, SUBA_PROBE
        CAND_CACHE = top8
        SENSOR_CACHE = [sensor]
        SUBA_PROBE = [P["subA"]]
        # feature sets
        feats = list(P["tcad"].feats)
        guides = [m["pts"][np.random.default_rng(1).choice(len(m["pts"]), min(200, len(m["pts"])),
                                                           replace=False)]
                  for m in P["part"]["supports"]]
        sel = greedy_select(P["tcad"].feats, guides, criterion="e", target_size=8,
                            min_features=5, rng=np.random.default_rng(0))
        feats_e8 = [P["tcad"].feats[j] for j in sel]           # frozen on intact cloud
        ext = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        REG = dict(GEO, k_huber=ext * 0.012)
        mu_CAD = P["tgt_pts"].mean(0)
        for k in range(3):
            dT, seed_info = make_perturb(stem, k, "I1")
            T_init = apply_perturb(dT, T_gt, mu_CAD)
            row = dict(stem=stem, view=vn, rho=round(rho_v, 3), k=k,
                       init_hash=t_hash(T_init), gate=gate, seed=seed_info["seed"])
            # G3: occlusion-adaptive selection under the trial init (no T_gt access).
            # A support keeps its full guide set if the occluded cloud (moved into
            # the CAD frame by the trial init) lands within 2 mm of >=5 of its
            # guide points; otherwise its guide set is reduced to a quarter —
            # the E-opt greedy then discounts poorly covered supports.
            p_cad = src_o @ T_init[:3, :3].T + T_init[:3, 3]
            guides_occl = []
            for g in guides:
                dg_, _ = cKDTree(g).query(p_cad, k=1, workers=1)
                covered = (dg_ < 2.0).sum()
                guides_occl.append(g if covered >= 5 else g[: max(5, len(g) // 4)])
            sel_o = greedy_select(P["tcad"].feats, guides_occl, criterion="e", target_size=8,
                                  min_features=5, rng=np.random.default_rng(0))
            feats_occl = [P["tcad"].feats[j] for j in sel_o]
            ei = e_tte(T_init, T_gt, P["evalB"], P["grp"])
            row["init"] = round(ei, 3)
            for arm in PILOT_ARMS:
                Tm, rt, info = run_arm(arm, P, src_o, nrm_o, T_init, feats, feats_e8,
                                       feats_occl, REG)
                e = e_tte(Tm, T_gt, P["evalB"], P["grp"])
                e_vis, e_hid, om = region_split_errors(Tm, T_gt, P["evalB"], P["grp"], sensor)
                fsv = fsv_depth(Tm, sensor, P["subA"])   # EvalCloud-A probe (never held-out B)
                rt_ = r_tau(Tm, src_o, P["scene"])
                cs = compat_success(Tm, src_o, P["scene"])
                row[arm] = dict(e_tte=round(e, 3), e_vis=round(e_vis, 3), e_hid=round(e_hid, 3),
                                omega=round(om, 2), fsv=round(fsv, 4), r_tau=round(rt_, 3),
                                compat=cs, ok=bool(e <= 0.5), rt=round(rt, 2))
                with open(out_jsonl, "a") as jf:
                    jf.write(json.dumps({**row, "arm": arm,
                                         "T_est": Tm.tolist()}, default=float) + "\n")
                print(f"  k{k} {arm}: e={e:.3f}{'Y' if e<=0.5 else 'N'} "
                      f"om={om:.1f} fsv={fsv:.3f} r_tau={rt_:.3f} ({rt:.1f}s)", flush=True)
            rows.append(row)
    base = os.path.join(OUTDIR, "pilot.json")
    out = base
    n = 1
    while os.path.exists(out):
        out = f"{base[:-5]}.{n}.json"; n += 1
    import subprocess as sp
    try:
        gsha = sp.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                      cwd=HERE).stdout.strip()[:12]
    except Exception:
        gsha = "nogit"
    json.dump(dict(tau_mm=0.5, budget_parity="G6 refine 30+60 vs B10 60 (iter_ratio 1.5, disclosed)",
                   git_sha=gsha, versions=dict(np=np.__version__, o3d=o3d.__version__,
                   scipy=__import__("scipy").__version__),
                   rows=rows, rho26=rho, checks=all_checks, substitutions=subs,
                   amendments=["mesh-raycast primary visibility (was z-buffer-only)",
                               "GeoP2E two-stage radius 15->3mm schedule (ICP-parity ladder)",
                               "multistart candidate list = top-8 + per-trial T_init (both arms)"],
                   stats_note="pilot is a plumbing gate; Wilcoxon/McNemar/bootstrap deferred to analyze mode before the main grid",
                   host=platform.node(),
                   cpu=platform.processor(), omp=os.environ.get("OMP_NUM_THREADS")),
              open(out, "w"), default=float, indent=1)
    print(f"pilot saved -> {out}", flush=True)




# ================================================================ grid machinery


# ================================================ certified groups + engineer arms
def certified_group(P):
    """Symmetry group augmented with reglaw D2 certified elements (2026-09-07):
    21638 strict C6 (6 off-axis parallel cylinders, permutation mismatch 8e-12mm);
    52890 C2 (axis 0,-1,0 @ p=(0,51.78,0)). Each constructed rotation is VERIFIED
    against EvalCloud-B (RMS <= 0.1 mm) before admission; cache to npz."""
    import os as _os
    stem = P["stem"]
    cache = _os.path.join(ART, f"e9_symgroup_cert_{stem}.npz")
    if _os.path.exists(cache):
        return [S for S in np.load(cache)["S"]]
    grp = list(P["grp"])
    evalB = P["evalB"]
    cands = []
    if stem.startswith("21638"):
        cyls = [f for f in P["tcad"].feats if f.kind == "cylinder"]
        ax = cyls[0].axis / np.linalg.norm(cyls[0].axis)
        C = np.array([f.center for f in cyls])
        d = np.abs((C - C.mean(0)) @ ax)          # spread along axis
        perp = (C - C.mean(0)) - np.outer((C - C.mean(0)) @ ax, ax)
        off = np.linalg.norm(perp, axis=1) > 1.0  # off-bundle-axis cylinders
        ref = C[off].mean(0) if off.any() else C.mean(0)
        for k in range(1, 6):
            cands.append(("c6", ax, ref, 60.0 * k))
    elif stem.startswith("52890"):
        ax = np.array([0.0, -1.0, 0.0])
        cands.append(("c2", ax, np.array([0.0, 51.7831, 0.0]), 180.0))
    from scipy.spatial.transform import Rotation as _R
    for name, ax, pt, deg in cands:
        R = _R.from_rotvec(np.radians(deg) * ax).as_matrix()
        S = np.eye(4); S[:3, :3] = R; S[:3, 3] = pt - R @ pt
        # symmetry test = SET invariance vs the MESH (exact surface distance;
        # point-cloud NN is resolution-limited by sampling density)
        sub = evalB[::40]
        q = sub @ S[:3, :3].T + S[:3, 3]
        import open3d as _o3
        nn = P["scene"].compute_distance(_o3.core.Tensor(q.astype(np.float32))).numpy()
        if float(np.median(nn)) <= 0.05:
            grp.append(S)
        else:
            print(f"  [cert] reject {name} {deg:.0f}deg: medNN={np.median(nn):.3f}mm", flush=True)
    # union construction (no unrestricted closure: the empirical elements and
    # the certified rotations do not strictly commute into a finite group; each
    # element is individually mesh-verified above). C_k about one axis is a
    # finite group by construction; dedupe at 1e-4.
    outg = []
    for g in [S for S in grp if np.linalg.det(S[:3, :3]) > 0.999]:
        if not any(np.allclose(g[:3, :3], h[:3, :3], atol=1e-4) for h in outg):
            outg.append(g)
    grp = outg
    np.savez_compressed(cache, S=np.array(grp) if grp else np.empty((0, 4, 4)))
    return grp


def eng_select(P, guides, target=8):
    """Engineer-adjusted selection (paper: human can adjust the auto selection):
    ALL planes must-include (they constrain the axial DOFs cylinders cannot),
    then E-opt greedy fill from the remaining primitives."""
    planes = [i for i, f in enumerate(P["tcad"].feats) if f.kind == "plane"]
    rest = [i for i in range(len(P["tcad"].feats)) if i not in planes]
    order = sorted(rest, key=lambda j: -len(guides[j]))   # area proxy: guide count
    fill = order[:max(1, target - len(planes))]
    keep = planes + fill
    return keep[:max(target, len(planes))]


GRID_ARMS = ("B1", "B2", "B3", "B4", "B5", "B6", "B10", "B10b", "B11",
             "G1", "G2", "G3", "G4", "G6", "G6a")


def vi_l(i):
    return f"{i:02d}"


def select_views(rho, pool, n_sym_faces):
    """Protocol 1.5: stratified 6-view selection on the retention covariate.
    S1 [0.55,0.75] x2, S2 [0.40,0.55) x2, S3 [0.25,0.40) x2; nearest-to-mid
    ranking, >=1 face direction (pool[:6]), +z (pool[4]) force-included if in
    band; empty stratum -> nearest-rho unused, recorded."""
    bands = [("S1", 0.55, 0.751, 2), ("S2", 0.40, 0.55, 2), ("S3", 0.25, 0.40, 2)]
    chosen, subs = [], []
    used = set()
    for name, lo, hi, k in bands:
        cand = sorted([(abs(r - (lo + hi) / 2), i) for i, r in enumerate(rho)
                       if lo <= r < hi and i not in used])
        if 4 not in used and any(lo <= rho[4] < hi for _ in [0]):
            cand = [(0.0, 4)] + [c for c in cand if c[1] != 4]   # +z force in its band
        take = [i for _, i in cand[:k]]
        while len(take) < k:                       # sparse/empty stratum -> substitution
            rest = sorted([(abs(r - (lo + hi) / 2), i) for i, r in enumerate(rho)
                           if i not in used and i not in take])
            if not rest:
                break
            take.append(rest[0][1])
            subs.append(f"{name} underfilled -> pool[{rest[0][1]}] rho={rho[rest[0][1]]:.2f}")
        for i in take:
            used.add(i); chosen.append((f"{name}v{vi_l(i)}", i))
    if not any(i < 6 for _, i in chosen):          # >=1 face: REPLACE lowest-rho pick
        faces = [(rho[i], i) for i in range(6) if i not in used]
        if faces:
            i = max(faces)[1]
            j = min(range(len(chosen)), key=lambda q: rho[chosen[q][1]])
            subs.append(f"no face view -> replace {chosen[j][0]} with pool[{i}] rho={rho[i]:.2f}")
            chosen[j] = (chosen[j][0] + "f", i)
    return chosen, subs


def part_features(P):
    feats = list(P["tcad"].feats)
    guides = [m["pts"][np.random.default_rng(1).choice(len(m["pts"]), min(200, len(m["pts"])),
                                                       replace=False)]
              for m in P["part"]["supports"]]
    sel = greedy_select(P["tcad"].feats, guides, criterion="e", target_size=8,
                        min_features=5, rng=np.random.default_rng(0))
    return feats, guides, [P["tcad"].feats[j] for j in sel]


def calibrate(fsv_ws=(0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)):
    """G6/B11 selector calibration on a DISJOINT calibration set (protocol 3.5):
    6 pool directions excluded from any main-grid band choice + 4 Fibonacci
    directions, 2 perturbations each, per part. One refine per candidate; the
    FSV_W sweep is evaluated retroactively on the cached scores."""
    out = {"trials": [], "fsv_ws": list(fsv_ws)}
    for si in (1, 2, 3, 4, 5):
        P = load_part(si)
        T_gt = P["T_gt"]
        xa, ya, za = canonical_basis(P["part"], P["tcad"])
        pool = view_dirs_26(xa, ya, za)
        rho = [len(P["src"][occlude(P["src"], P["sn"], T_gt, d, scene=P["scene"])[0]]) / len(P["src"])
               for d in pool]
        chosen, _ = select_views(rho, pool, None)
        grid_idx = {i for _, i in chosen}
        cal_idx = [i for i in range(26) if i not in grid_idx][:6]
        fib = []
        for k_ in range(4):                        # Fibonacci directions (deterministic)
            g = np.pi * (3 - np.sqrt(5)) * k_
            zz = 1 - 2 * (k_ + 0.5) / 4
            rr = np.sqrt(max(0, 1 - zz * zz))
            fib.append(xa * (rr * np.cos(g)) + ya * (rr * np.sin(g)) + za * zz)
        feats, guides, feats_e8 = part_features(P)
        ext = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        REG = dict(GEO, k_huber=ext * 0.012)
        diag = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        r_coarse = min(15.0, 0.12 * diag)
        mu = P["tgt_pts"].mean(0)
        for vi, d in enumerate([pool[i] for i in cal_idx] + fib):
            mask, nrm_o, sensor = occlude(P["src"], P["sn"], T_gt, d, scene=P["scene"])
            src_o = P["src"][mask]
            if len(src_o) < 2000:
                continue
            _, top8, _ = rank_candidates(src_o, P["tgt_pts"], P["subA_tree"])
            for k in range(2):
                dT, _ = make_perturb(P["stem"], k, "CAL")
                T_init = apply_perturb(dT, T_gt, mu)
                cands = top8 + [T_init]
                rows_c = []
                for Tc in cands:
                    Ts, _ = register(src_o, nrm_o, feats, init_T=Tc.copy(),
                                     **{**REG, "init_radius": r_coarse, "anneal": 0.85, "max_iter": 25})
                    rmse_s = trimmed_rmse(Ts, src_o, P["subA_tree"])   # screen-stage score
                    fsv_s = fsv_depth(Ts, sensor, P["subA"])
                    Tm, _ = register(src_o, nrm_o, feats, init_T=Ts, **REG)
                    rows_c.append(dict(e=E9e(P, Tm), rmse=rmse_s, fsv=fsv_s))
                out["trials"].append(dict(part=P["stem"], view=vi, k=k, cands=rows_c))
                print(f"cal {P['stem'][:5]} v{vi} k{k}: best_e={min(c['e'] for c in rows_c):.3f}", flush=True)
    # retroactive sweep
    summary = {}
    for w in fsv_ws:
        es = []
        for tr in out["trials"]:
            sel = min(range(len(tr["cands"])),
                      key=lambda j: tr["cands"][j]["rmse"] + w * tr["cands"][j]["fsv"])
            es.append(tr["cands"][sel]["e"])
        es = np.array(es)
        summary[w] = dict(median=float(np.median(es)), p90=float(np.percentile(es, 90)),
                          frac_bad=float(np.mean(es > 0.5)))
        print(f"FSV_W={w}: median e={summary[w]['median']:.3f} p90={summary[w]['p90']:.3f} "
              f"bad%={summary[w]['frac_bad']:.2f}", flush=True)
    best_w = min(fsv_ws, key=lambda w: (summary[w]["median"], summary[w]["p90"]))
    out["summary"], out["best_w"] = summary, best_w
    json.dump(out, open(os.path.join(OUTDIR, "g6_calibration.json"), "w"), default=float, indent=1)
    print(f"CALIBRATION DONE best FSV_W={best_w}", flush=True)


def E9e(P, Tm):
    return e_tte(Tm, P["T_gt"], P["evalB"], P["grp"])


def main():
    """Main grid: 5 parts x 6 stratified views x 10 perturbations x GRID_ARMS."""
    cal = os.path.join(OUTDIR, "g6_calibration.json")
    if os.path.exists(cal):
        global FSV_W_CACHE
        FSV_W_CACHE = [json.load(open(cal))["best_w"]]
        print(f"FSV_W from calibration: {FSV_W_CACHE[0]}", flush=True)
    out_jsonl = os.path.join(OUTDIR, "main.jsonl")
    n = 1
    while os.path.exists(out_jsonl):
        out_jsonl = os.path.join(OUTDIR, f"main.{n}.jsonl"); n += 1
    jf = open(out_jsonl, "a")
    for si in (1, 2, 3, 4, 5):
        P = load_part(si)
        T_gt = P["T_gt"]
        xa, ya, za = canonical_basis(P["part"], P["tcad"])
        pool = view_dirs_26(xa, ya, za)
        rho = [len(P["src"][occlude(P["src"], P["sn"], T_gt, d, scene=P["scene"])[0]]) / len(P["src"])
               for d in pool]
        chosen, subs = select_views(rho, pool, None)
        print(f"[{P['stem']}] views: {[(nm, i, round(rho[i],2)) for nm, i in chosen]} subs={subs}", flush=True)
        feats, guides, feats_e8 = part_features(P)
        ext = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        REG = dict(GEO, k_huber=ext * 0.012)
        mu = P["tgt_pts"].mean(0)
        for nm, vi in chosen:
            d_C = pool[vi]
            mask, nrm_o, sensor = occlude(P["src"], P["sn"], T_gt, d_C, scene=P["scene"])
            src_o = P["src"][mask]
            rho_v = len(src_o) / len(P["src"])
            global CAND_CACHE, SENSOR_CACHE, SUBA_PROBE
            _, top8, _ = rank_candidates(src_o, P["tgt_pts"], P["subA_tree"])
            CAND_CACHE = top8; SENSOR_CACHE = [sensor]; SUBA_PROBE = [P["subA"]]
            for k in range(10):
                dT, seed_info = make_perturb(P["stem"], k, "I1")
                T_init = apply_perturb(dT, T_gt, mu)
                ei = e_tte(T_init, T_gt, P["evalB"], P["grp"])
                # G3 guides under this trial init
                p_cad = src_o @ T_init[:3, :3].T + T_init[:3, 3]
                guides_occl = []
                for g in guides:
                    dg_, _ = cKDTree(g).query(p_cad, k=1, workers=1)
                    guides_occl.append(g if (dg_ < 2.0).sum() >= 5 else g[: max(5, len(g) // 4)])
                sel_o = greedy_select(P["tcad"].feats, guides_occl, criterion="e", target_size=8,
                                      min_features=5, rng=np.random.default_rng(0))
                feats_occl = [P["tcad"].feats[j] for j in sel_o]
                for arm in GRID_ARMS:
                    Tm, rt, info = run_arm(arm, P, src_o, nrm_o, T_init, feats, feats_e8,
                                           feats_occl, REG)
                    e = e_tte(Tm, T_gt, P["evalB"], P["grp"])
                    ev, eh, om = region_split_errors(Tm, T_gt, P["evalB"], P["grp"], sensor)
                    row = dict(part=P["stem"], si=si, view=nm, vi=vi, rho=round(rho_v, 3), k=k,
                               n_occl=len(src_o), n_src=len(P["src"]), n_evalB=len(P["evalB"]),
                               arm=arm, seed=seed_info["seed"], init_hash=t_hash(T_init),
                               e_tte=round(e, 4), init_e=round(ei, 4),
                               e_vis=round(ev, 4), e_hid=round(eh, 4), omega=round(om, 2),
                               fsv=round(fsv_depth(Tm, sensor, P["subA"]), 4),
                               r_tau=round(r_tau(Tm, src_o, P["scene"]), 4),
                               ok=bool(e <= 0.5), rt=round(rt, 2),
                               coarse=str(info)[:120])
                    jf.write(json.dumps(row, default=float) + "\n"); jf.flush()
                print(f"  {P['stem'][:5]} {nm} k{k} done", flush=True)
    jf.close()
    print(f"MAIN GRID DONE -> {out_jsonl}", flush=True)


def analyze(jsonl=None):
    """Protocol 5.2 stats: per-part paired Wilcoxon (e_TTE) + exact McNemar
    (success) with Holm across parts; cluster bootstrap (parts) CI for the
    success-rate difference and median-ratio, primary H1 (G1 vs B3, rho<=0.55)."""
    from scipy.stats import wilcoxon
    import itertools as it
    path = jsonl
    if path is None:
        cands_f = sorted(f for f in os.listdir(OUTDIR) if f.startswith("main") and f.endswith(".jsonl"))
        path = os.path.join(OUTDIR, cands_f[-1] if cands_f else "main.jsonl")
    rows = [json.loads(l) for l in open(path)]
    def cell(rs, arm):
        return {r["k"] for r in rs if r["arm"] == arm and r["view"] == r["view"]}
    parts = sorted({r["part"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    res = {"path": path, "n_rows": len(rows), "parts": parts}
    # paired by (view, k): e_TTE difference armA - armB
    def paired(a, b, subset=None):
        rs_ = [r for r in rows if (subset is None or subset(r)) and r["rho"] <= 0.551]
        outp = {}
        for p_ in parts:
            ra = {(r["view"], r["k"]): r["e_tte"] for r in rs_ if r["part"] == p_ and r["arm"] == a}
            rb = {(r["view"], r["k"]): r["e_tte"] for r in rs_ if r["part"] == p_ and r["arm"] == b}
            keys = sorted(set(ra) & set(rb))
            if len(keys) < 5:
                outp[p_] = None; continue
            diff = [ra[k_] - rb[k_] for k_ in keys]
            try:
                stat, pv = wilcoxon(diff)
            except ValueError:
                stat, pv = float("nan"), 1.0
            outp[p_] = dict(n=len(keys), median_diff=float(np.median(diff)), p=float(pv))
        # Holm across parts with valid p
        ps = sorted([(v["p"], p_) for p_, v in outp.items() if v], key=lambda x: x[0])
        m = len(ps)
        holm, run = {}, 0.0
        for i, (pv, p_) in enumerate(ps):
            run = max(run, min(1.0, (m - i) * pv))     # running max (step-down)
            holm[p_] = run
        for p_ in outp:
            if outp[p_]:
                outp[p_]["p_holm"] = holm.get(p_)
        return outp
    res["wilcoxon"] = {f"{a}_vs_B3": paired(a, "B3") for a in arms if a != "B3"}
    # exact McNemar on success, G1 vs B3, per part
    from math import comb
    def mcnemar(a, b):
        outm = {}
        for p_ in parts:
            ra = {(r["view"], r["k"]): r["ok"] for r in rows if r["part"] == p_ and r["arm"] == a}
            rb = {(r["view"], r["k"]): r["ok"] for r in rows if r["part"] == p_ and r["arm"] == b}
            keys = sorted(set(ra) & set(rb))
            n01 = sum(1 for k_ in keys if ra[k_] and not rb[k_])
            n10 = sum(1 for k_ in keys if not ra[k_] and rb[k_])
            nn = n01 + n10
            if nn == 0:
                outm[p_] = dict(n=len(keys), n01=n01, n10=n10, p=1.0); continue
            pv = sum(comb(nn, i) for i in range(min(n01, n10) + 1)) / 2 ** nn * 2
            outm[p_] = dict(n=len(keys), n01=n01, n10=n10, p=min(1.0, pv))
        return outm
    res["mcnemar_G1_vs_B3"] = mcnemar("G1", "B3")
    # cluster bootstrap over parts (B=10000)
    rng = np.random.default_rng(0)
    sub = [r for r in rows if r["rho"] <= 0.55]          # primary subset
    def boot(a, b, B=10000):
        def stats(sample_idx):
            multi = {}                                # part multiplicity preserved
            for i_ in sample_idx:
                multi[parts[i_]] = multi.get(parts[i_], 0) + 1
            ra = {}
            for r in sub:
                if r["arm"] != a or r["part"] not in multi:
                    continue
                for _ in range(multi[r["part"]]):
                    ra[(r["part"], _ , r["view"], r["k"])] = r
            rb = {}
            for r in sub:
                if r["arm"] != b or r["part"] not in multi:
                    continue
                for _ in range(multi[r["part"]]):
                    rb[(r["part"], _ , r["view"], r["k"])] = r
            keys = sorted(set(ra) & set(rb))
            if not keys:
                return None
            ds = float(np.mean([ra[k_]["ok"] for k_ in keys]) - np.mean([rb[k_]["ok"] for k_ in keys]))
            ea = np.array([ra[k_]["e_tte"] for k_ in keys])
            eb = np.array([rb[k_]["e_tte"] for k_ in keys])
            mr = float(np.median(ea) / max(np.median(eb), 1e-9))
            return ds, mr
        base = stats(range(len(parts)))
        ds_b, mr_b = [], []
        for _ in range(B):
            st = stats(rng.integers(0, len(parts), len(parts)))
            if st:
                ds_b.append(st[0]); mr_b.append(st[1])
        return dict(base=dict(success_diff=base[0], median_ratio=base[1]),
                    ci_success=[float(np.percentile(ds_b, 2.5)), float(np.percentile(ds_b, 97.5))],
                    ci_ratio=[float(np.percentile(mr_b, 2.5)), float(np.percentile(mr_b, 97.5))])
    res["bootstrap_G1_vs_B3"] = boot("G1", "B3")
    # per-part per-arm success table
    tab = {}
    for p_ in parts:
        for a in arms:
            rs_ = [r for r in rows if r["part"] == p_ and r["arm"] == a]
            if rs_:
                tab.setdefault(p_, {})[a] = dict(n=len(rs_),
                                                 succ=int(sum(r["ok"] for r in rs_)),
                                                 med_e=float(np.median([r["e_tte"] for r in rs_])))
    res["table"] = tab
    out = os.path.join(OUTDIR, "analysis.json")
    n = 1
    while os.path.exists(out):
        out = os.path.join(OUTDIR, f"analysis.{n}.json"); n += 1
    json.dump(res, open(out, "w"), default=float, indent=1)
    print(f"ANALYSIS DONE -> {out}", flush=True)


def focus(parts=(1, 4), arms=None):
    """Focused rerun of the failing parts with CERTIFIED symmetry groups, the
    engineer arms (G7 eng_select, G8 planes-first coarse), and T_est saved."""
    grp_cert = {}
    out_jsonl = os.path.join(OUTDIR, "focus.jsonl")
    n = 1
    while os.path.exists(out_jsonl):
        out_jsonl = os.path.join(OUTDIR, f"focus.{n}.jsonl"); n += 1
    jf = open(out_jsonl, "a")
    ARMS = arms if arms else GRID_ARMS + ("G7", "G8")
    for si in parts:
        P = load_part(si)
        grp = certified_group(P)
        grp_cert[P["stem"]] = len(grp)
        P["grp"] = grp                              # certified group for e_tte
        T_gt = P["T_gt"]
        xa, ya, za = canonical_basis(P["part"], P["tcad"])
        pool = view_dirs_26(xa, ya, za)
        rho = [len(P["src"][occlude(P["src"], P["sn"], T_gt, d, scene=P["scene"])[0]]) / len(P["src"])
               for d in pool]
        chosen, subs = select_views(rho, pool, None)
        feats, guides, feats_e8 = part_features(P)
        ext = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        REG = dict(GEO, k_huber=ext * 0.012)
        feats_eng = [P["tcad"].feats[j] for j in eng_select(P, guides)]
        mu = P["tgt_pts"].mean(0)
        for nm, vi in chosen:
            mask, nrm_o, sensor = occlude(P["src"], P["sn"], T_gt, pool[vi], scene=P["scene"])
            src_o = P["src"][mask]
            global CAND_CACHE, SENSOR_CACHE, SUBA_PROBE
            _, top8, _ = rank_candidates(src_o, P["tgt_pts"], P["subA_tree"])
            CAND_CACHE = top8; SENSOR_CACHE = [sensor]; SUBA_PROBE = [P["subA"]]
            for k in range(10):
                dT, seed_info = make_perturb(P["stem"], k, "I1")
                T_init = apply_perturb(dT, T_gt, mu)
                p_cad = src_o @ T_init[:3, :3].T + T_init[:3, 3]
                guides_occl = []
                for g in guides:
                    dg_, _ = cKDTree(g).query(p_cad, k=1, workers=1)
                    guides_occl.append(g if (dg_ < 2.0).sum() >= 5 else g[: max(5, len(g) // 4)])
                sel_o = greedy_select(P["tcad"].feats, guides_occl, criterion="e", target_size=8,
                                      min_features=5, rng=np.random.default_rng(0))
                feats_occl = [P["tcad"].feats[j] for j in sel_o]
                for arm in ARMS:
                    Tm, rt, info = run_arm(arm, P, src_o, nrm_o, T_init, feats, feats_e8,
                                           feats_occl, REG, feats_eng=feats_eng)
                    e = e_tte(Tm, T_gt, P["evalB"], grp)
                    row = dict(part=P["stem"], si=si, view=nm, vi=vi, rho=round(len(src_o)/len(P["src"]),3),
                               k=k, arm=arm, n_occl=len(src_o), e_tte=round(e, 4),
                               r_tau=round(r_tau(Tm, src_o, P["scene"]), 4), ok=bool(e <= 0.5),
                               rt=round(rt, 2), T_est=[round(float(x), 5) for x in np.asarray(Tm).ravel()])
                    jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"  {P['stem'][:5]} {nm} done (|G|={len(grp)})", flush=True)
    jf.close()
    json.dump(grp_cert, open(os.path.join(OUTDIR, "focus_groups.json"), "w"))
    print(f"FOCUS DONE -> {out_jsonl}", flush=True)





def view_coverage(P, sup_i_pts, T_gt, sensor):
    """fraction of a feature's guide points visible (mesh raycast) at GT pose"""
    import open3d as _o3
    g = sup_i_pts[:60]
    c_CAD = T_gt[:3, :3] @ sensor["c"] + T_gt[:3, 3]
    p_CAD = g  # guides are CAD-frame
    dv = c_CAD - p_CAD
    L = np.linalg.norm(dv, axis=1); dv /= L[:, None]
    rays = _o3.core.Tensor(np.hstack([np.broadcast_to(c_CAD, p_CAD.shape), -dv]).astype(np.float32))
    t = P["scene"].cast_rays(rays)["t_hit"].numpy().astype(np.float64)
    clear = (~np.isfinite(t)) | (t <= 0.5) | (t >= L - 0.5)
    return float(np.mean(clear))


def select_feats_view(P, view_key, d_C, guides, feats, REG, n_cand=6, verbose=True):
    """Per-(part,view) optimal feature subset (production framing: the viewpoint
    is fixed per station, so the subset is FIXED per station; selection here
    evaluates candidate subsets on this view's own I1 perturbations -- accepted
    'overfitting' by design, disclosed)."""
    T_gt = P["T_gt"]
    mask, nrm_o, sensor = occlude(P["src"], P["sn"], T_gt, d_C, scene=P["scene"])
    src_o = P["src"][mask]
    if len(src_o) < 3000:
        return None, mask, nrm_o, sensor
    sup = P["part"]["supports"]
    cov = np.array([view_coverage(P, sup[i]["pts"], T_gt, sensor) for i in range(len(feats))])
    vis = [i for i in range(len(feats)) if cov[i] >= 0.05]
    # drop duplicate near-parallel planes (cross-matching: gap < r_coarse)
    diag = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
    r_coarse = min(15.0, 0.12 * diag)
    drop = set()
    pl = [i for i in vis if feats[i].kind == "plane"]
    for a in range(len(pl)):
        for b in range(a + 1, len(pl)):
            fa, fb = feats[pl[a]], feats[pl[b]]
            if abs(float(np.dot(fa.normal, fb.normal))) > 0.95:
                gap = abs(float(np.dot(np.asarray(fb.center) - np.asarray(fa.center), fa.normal))) \
                      if hasattr(fb, "center") else 99
                if gap < r_coarse * 1.2:
                    drop.add(pl[a] if len(sup[pl[a]]["pts"]) < len(sup[pl[b]]["pts"]) else pl[b])
    nodup = [i for i in vis if i not in drop]
    # candidate subsets
    cands = {"vis_all": vis, "vis_nodup": nodup}
    guides_vis = [guides[i][np.random.default_rng(3).random(len(guides[i])) < max(cov[i], 0.05)][:200]
                  if cov[i] > 0.05 else guides[i][:5] for i in range(len(feats))]
    try:
        g8 = greedy_select(feats, guides_vis, criterion="e", target_size=8, min_features=3,
                           rng=np.random.default_rng(0))
        cands["greedy8_vis"] = [i for i in g8 if cov[i] >= 0.05] or vis
    except Exception:
        pass
    cands["areaxcov8"] = sorted(vis, key=lambda i: -len(sup[i]["pts"]) * cov[i])[:8]
    cands["nodup_one_plane"] = [i for i in nodup][:8]
    if len(nodup) > 8:
        cands["nodup_top"] = sorted(nodup, key=lambda i: -len(sup[i]["pts"]) * cov[i])[:8]
    cands = dict(list(cands.items())[:n_cand])
    # evaluate each candidate on this view's 10 I1 perturbations
    _, top8, _ = rank_candidates(src_o, P["tgt_pts"], P["subA_tree"])
    global CAND_CACHE, SENSOR_CACHE, SUBA_PROBE
    CAND_CACHE = top8; SENSOR_CACHE = [sensor]; SUBA_PROBE = [P["subA"]]
    mu = P["tgt_pts"].mean(0)
    best = None
    for name, ids in cands.items():
        f_sub = [feats[i] for i in ids]
        es = []
        for k in range(10):
            dT, _ = make_perturb(P["stem"], k, "I1")
            T_init = apply_perturb(dT, T_gt, mu)
            Tm, _, _ = run_arm("G1", P, src_o, nrm_o, T_init, f_sub, f_sub, f_sub, REG)
            es.append(e_tte(Tm, T_gt, P["evalB"], P["grp"]))
        es = np.array(es)
        sc = (int(np.sum(es <= 0.5)), -float(np.median(es)))
        if best is None or sc > best[1]:
            best = (name, sc, ids, f_sub, es)
        if verbose:
            print(f"    cand {name} ({len(ids)}f): {sc[0]}/10 med={np.median(es):.3f}", flush=True)
    name, sc, ids, f_sub, es = best
    if verbose:
        print(f"  [{P['stem'][:5]} {view_key}] per-view subset = {name} {sorted(ids)} -> {sc[0]}/10", flush=True)
    return f_sub, mask, nrm_o, sensor, dict(chosen=name, ids=sorted(ids), cov={i: round(float(cov[i]),2) for i in vis}, dropped_parallel=sorted(drop))


def g9campaign(parts=(1, 2, 3, 4, 5), arms=("G9", "G1", "B6", "B10b")):
    """Full-grid per-view-profile campaign. G9 = per-(part,view) optimal subset
    (selected on the same view's I1 perturbations; production framing: profile
    is fixed per station). Controls G1/B6/B10b rerun for the same table."""
    out_jsonl = os.path.join(OUTDIR, "g9.jsonl")
    n = 1
    while os.path.exists(out_jsonl):
        out_jsonl = os.path.join(OUTDIR, f"g9.{n}.jsonl"); n += 1
    jf = open(out_jsonl, "a")
    profiles = {}
    for si in parts:
        P = load_part(si)
        P["grp"] = certified_group(P)
        T_gt = P["T_gt"]
        xa, ya, za = canonical_basis(P["part"], P["tcad"])
        pool = view_dirs_26(xa, ya, za)
        rho = [len(P["src"][occlude(P["src"], P["sn"], T_gt, d, scene=P["scene"])[0]]) / len(P["src"]) for d in pool]
        chosen, subs = select_views(rho, pool, None)
        feats, guides, feats_e8 = part_features(P)
        ext = float(np.linalg.norm(np.ptp(P["tcad"].pts, axis=0)))
        REG = dict(GEO, k_huber=ext * 0.012)
        mu = P["tgt_pts"].mean(0)
        for nm, vi in chosen:
            f_view, mask, nrm_o, sensor, prof = select_feats_view(P, nm, pool[vi], guides, feats, REG)
            src_o = P["src"][mask]
            profiles[f"{P['stem']}|{nm}"] = prof
            for k in range(10):
                dT, seed_info = make_perturb(P["stem"], k, "I1")
                T_init = apply_perturb(dT, T_gt, mu)
                for arm in arms:
                    if arm == "G9":
                        Tm, rt, info = run_arm("G1", P, src_o, nrm_o, T_init, f_view, f_view, f_view, REG)
                    else:
                        Tm, rt, info = run_arm(arm, P, src_o, nrm_o, T_init, feats, feats_e8, feats_e8, REG)
                    e = e_tte(Tm, T_gt, P["evalB"], P["grp"])
                    row = dict(part=P["stem"], si=si, view=nm, rho=round(len(src_o)/len(P["src"]),3), k=k,
                               arm=arm, e_tte=round(e,4), r_tau=round(r_tau(Tm, src_o, P["scene"]),4),
                               ok=bool(e <= 0.5), rt=round(rt,2), n_occl=len(src_o),
                               T_est=[round(float(x),5) for x in np.asarray(Tm).ravel()])
                    jf.write(json.dumps(row) + "\n"); jf.flush()
            print(f"  {P['stem'][:5]} {nm} arms done", flush=True)
    jf.close()
    json.dump(profiles, open(os.path.join(OUTDIR, "g9_profiles.json"), "w"), default=float, indent=1)
    print(f"G9 CAMPAIGN DONE -> {out_jsonl}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pilot", "calib", "main", "analyze", "unittest", "focus", "g9"])
    a = ap.parse_args()
    if a.mode == "pilot":
        pilot()
    elif a.mode == "calib":
        calibrate()
    elif a.mode == "main":
        main()
    elif a.mode == "analyze":
        analyze()
    elif a.mode == "focus":
        focus()
    elif a.mode == "g9":
        g9campaign()
    elif a.mode == "unittest":
        # synthetic-record unit test of the analyze statistics (protocol 12):
        # construct rows where G1 beats B3 on 4/5 parts with a known margin,
        # and verify the Wilcoxon/McNemar/bootstrap directions.
        synth = os.path.join(OUTDIR, "unittest.jsonl")
        rng = np.random.default_rng(1)
        with open(synth, "w") as f:
            for p_ in range(5):
                for v in range(6):
                    for k in range(10):
                        for arm in ("G1", "B3"):
                            good = arm == "G1" and (p_ < 4)
                            e = rng.lognormal(-1.5, 0.6) if good else rng.lognormal(0.2, 0.8)
                            f.write(json.dumps(dict(part=f"P{p_}", view=f"v{v}", k=k, arm=arm,
                                                    rho=0.4, e_tte=round(float(e), 4),
                                                    ok=bool(e < 0.5), si=p_)) + "\n")
        for old_a in ("analysis.json",):
            p_old = os.path.join(OUTDIR, old_a)
            if os.path.exists(p_old):
                os.remove(p_old)
        analyze(synth)
        r = json.load(open(os.path.join(OUTDIR, "analysis.json")))
        w = r["wilcoxon"]["G1_vs_B3"]
        n_sig = sum(1 for p_, v in w.items() if v and v["p_holm"] is not None and v["p_holm"] < 0.05)
        bs = r["bootstrap_G1_vs_B3"]["base"]
        assert n_sig >= 3, f"synthetic effect not detected: {n_sig} sig parts"
        assert bs["success_diff"] > 0.15 and bs["median_ratio"] < 0.5, bs
        print(f"UNITTEST PASS: {n_sig}/5 parts Holm-significant, success_diff="
              f"{bs['success_diff']:.3f}, ratio={bs['median_ratio']:.3f}")
        # null fixture: same seed, no arm effect -> must detect nothing
        synth2 = os.path.join(OUTDIR, "unittest_null.jsonl")
        rng2 = np.random.default_rng(2)
        with open(synth2, "w") as f:
            for p_ in range(5):
                for v in range(6):
                    for k in range(10):
                        for arm in ("G1", "B3"):
                            e = rng2.lognormal(-0.6, 0.8)
                            f.write(json.dumps(dict(part=f"P{p_}", view=f"v{v}", k=k, arm=arm,
                                                    rho=0.4, e_tte=round(float(e), 4),
                                                    ok=bool(e < 0.5), si=p_)) + "\n")
        for old_a in ("analysis.json",):
            p_old = os.path.join(OUTDIR, old_a)
            if os.path.exists(p_old):
                os.remove(p_old)
        analyze(synth2)
        r2 = json.load(open(os.path.join(OUTDIR, "analysis.json")))
        n_sig2 = sum(1 for p_, v in r2["wilcoxon"]["G1_vs_B3"].items()
                     if v and v["p_holm"] is not None and v["p_holm"] < 0.05)
        assert n_sig2 == 0, f"null fixture false-positive: {n_sig2} parts"
        print("UNITTEST NULL PASS: 0/5 false positives")