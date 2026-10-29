"""Extended gear-baseline experiment: adds Sparse ICP (Bouaziz-style L_p ADMM) and
runs all baselines on 4 real gear scans (3 defective + 1 complete). Point-to-mesh
residual (RaycastingScene) vs GeoP2E published 0.25-0.28 mm. Runs in geo env."""
import sys, os, json
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import trimmed_icp

GEAR = "data/gear_step_cad"
SCANS = [("defect/22.pcd", "defect ~30% loss"), ("defect/4.ply", "defective 4"),
         ("defect/1.ply", "fragment 1"), ("scan_model.obj", "complete scan")]


def fpfh(p, voxel):
    d = p.voxel_down_sample(voxel)
    d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return d, f


def sparse_icp(source, target, init, threshold=2.0, p=0.5, outer=20, irls=4):
    """L_p (p<1) Sparse ICP via IRLS (Bouaziz et al. 2013, iteratively-reweighted form).
    Robust to outliers via L_p sparsity-promoting per-correspondence weights; the rigid pose
    is solved by weighted Kabsch each IRLS step (absolute source->target, so T is set, not
    composed). From a coarse init (FGR)."""
    tree = cKDTree(target)
    R = init[:3, :3].copy(); t = init[:3, 3].copy()
    for _ in range(outer):
        cur = source @ R.T + t
        d, idx = tree.query(cur, k=1); mask = d < threshold
        if mask.sum() < 10:
            break
        P = source[mask]; Q = target[idx[mask]]
        for _ in range(irls):
            r = np.linalg.norm((P @ R.T + t) - Q, axis=1)
            w = 1.0 / np.maximum(r, 1e-9) ** (2 - p)          # L_p IRLS weights
            ws = w.sum()
            cP = (P * w[:, None]).sum(0) / ws; cQ = (Q * w[:, None]).sum(0) / ws
            H = ((P - cP) * w[:, None]).T @ (Q - cQ)
            U, S, Vt = np.linalg.svd(H); Rm = Vt.T @ U.T
            if np.linalg.det(Rm) < 0:
                Vt[2] *= -1; Rm = Vt.T @ U.T
            t = cQ - Rm @ cP; R = Rm
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t
    return T


def main():
    cad_mesh = o3d.io.read_triangle_mesh(f"{GEAR}/gear/model.obj"); cad_mesh.compute_vertex_normals()
    cad_pc = cad_mesh.sample_points_uniformly(number_of_points=20000)
    cad_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    cadpts = np.asarray(cad_pc.points)
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cad_mesh))

    def p2m(pts):
        return scene.compute_distance(o3d.core.Tensor(np.asarray(pts, dtype=np.float32))).numpy()

    _, ft = fpfh(cad_pc, 1.0); td = cad_pc.voxel_down_sample(1.0)
    rows = []
    for sf, label in SCANS:
        path = f"{GEAR}/{sf}"
        if not os.path.exists(path):
            print(f"skip {sf}"); continue
        if sf.endswith(".obj"):
            m = o3d.io.read_triangle_mesh(path); src_raw = o3d.geometry.PointCloud(); src_raw.points = m.vertices
        else:
            src_raw = o3d.io.read_point_cloud(path)
        if len(src_raw.points) < 100:
            import trimesh; src_raw.points = o3d.utility.Vector3dVector(np.asarray(trimesh.load(path).vertices))
        pts = np.asarray(src_raw.points)
        src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(pts)
        src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        sd, fs = fpfh(src_raw, 1.0)
        opt = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=1.5, use_absolute_scale=True)
        try:
            T_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(sd, td, fs, ft, opt).transformation
        except Exception as e:
            print(f"{sf}: FGR failed {e}"); continue

        def icp(est, thr=2.0):
            return o3d.pipelines.registration.registration_icp(src, cad_pc, thr, T_fgr, est).transformation
        res = {}
        for name, T in [("FGR", T_fgr),
                        ("FGR+ICP-p2p", icp(o3d.pipelines.registration.TransformationEstimationPointToPoint())),
                        ("FGR+ICP-p2plane", icp(o3d.pipelines.registration.TransformationEstimationPointToPlane())),
                        ("FGR+GICP", o3d.pipelines.registration.registration_generalized_icp(src, cad_pc, 2.0, T_fgr, o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()).transformation),
                        ("FGR+Trimmed", trimmed_icp(pts, cadpts, init=T_fgr, threshold=2.0, trim_frac=0.3, max_iter=50)[0]),
                        ("FGR+SparseICP", sparse_icp(pts, cadpts, init=T_fgr, threshold=3.0, p=0.5, outer=20))]:
            d = p2m(pts @ T[:3, :3].T + T[:3, 3]); res[name] = (float(np.median(d)), float(np.mean(d < 2.0) * 100))
        rows.append({"scan": sf, "label": label, "n_pts": len(pts), "results": res})
        print(f"\n{sf} ({label}, {len(pts)} pts) — GeoP2E published 0.25-0.28 mm")
        for mname, (med, inl) in res.items():
            print(f"  {mname:16s} resid_med={med:.3f}mm inlier(<2mm)={inl:.0f}%")
    out = {"metric": "point-to-MESH median mm (RaycastingScene); GeoP2E published 0.25-0.28 mm (cited)",
           "scans": rows}
    json.dump(out, open("results/gear_baselines_extended.json", "w"), indent=2)
    print("\nSaved results/gear_baselines_extended.json")


if __name__ == "__main__":
    main()
