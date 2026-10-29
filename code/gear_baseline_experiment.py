"""Real-gear defect BASELINE comparison (R3-5) — runs in geo env (open3d + scipy).

Uses the REAL gear CAD + REAL defective scans. Runs the baselines the reviewers
requested (FGR global, +ICP point-to-point, +ICP point-to-plane, +GICP, Trimmed ICP)
and measures the point-to-CAD residual. Compares to GeoP2E's PUBLISHED result
(Table 3: 0.25 mm @30%, 0.28 mm @50% volume loss) — which is cited, not re-run.
Baselines operate on point clouds (no need for the degenerate coaxial features).
"""
import sys, os, json
import numpy as np
import open3d as o3d
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import trimmed_icp

GEAR = "data/gear_step_cad"
SCANS = [("defect/22.pcd", "30%-loss gear (approx)"), ("defect/4.ply", "defective scan 4"), ("defect/1.ply", "fragment scan 1")]


def fpfh(p, voxel):
    d = p.voxel_down_sample(voxel)
    d.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2, max_nn=30))
    f = o3d.pipelines.registration.compute_fpfh_feature(d, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))
    return d, f


def make_p2m(cad_mesh):
    """point-to-MESH distance via RaycastingScene (same metric as GeoP2E Table 3)."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cad_mesh))
    def f(pts):
        return scene.compute_distance(o3d.core.Tensor(np.asarray(pts, dtype=np.float32))).numpy()
    return f


def main():
    cad_mesh = o3d.io.read_triangle_mesh(f"{GEAR}/gear/model.obj"); cad_mesh.compute_vertex_normals()
    cad_pc = cad_mesh.sample_points_uniformly(number_of_points=20000)
    cad_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
    cadpts = np.asarray(cad_pc.points); p2m = make_p2m(cad_mesh)
    _, ft = fpfh(cad_pc, 1.0)
    td = cad_pc.voxel_down_sample(1.0)

    rows = []
    for scan_file, label in SCANS:
        path = f"{GEAR}/{scan_file}"
        if not os.path.exists(path):
            continue
        src_raw = o3d.io.read_point_cloud(path)
        if len(src_raw.points) == 0:
            import trimesh; m = trimesh.load(path); src_raw.points = o3d.utility.Vector3dVector(np.asarray(m.vertices))
        pts = np.asarray(src_raw.points)
        src = o3d.geometry.PointCloud(); src.points = o3d.utility.Vector3dVector(pts)
        src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30))
        sd, fs = fpfh(src_raw, 1.0)
        opt = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=1.5, use_absolute_scale=True)
        try:
            fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(sd, td, fs, ft, opt)
            T_fgr = fgr.transformation
        except Exception as e:
            print(f"{scan_file}: FGR failed {e}"); continue

        def icp(est, thr=2.0):
            return o3d.pipelines.registration.registration_icp(src, cad_pc, thr, T_fgr, est).transformation

        methods = {
            "FGR (global)": T_fgr,
            "FGR+ICP-p2p": icp(o3d.pipelines.registration.TransformationEstimationPointToPoint()),
            "FGR+ICP-p2plane": icp(o3d.pipelines.registration.TransformationEstimationPointToPlane()),
            "FGR+GICP": o3d.pipelines.registration.registration_generalized_icp(src, cad_pc, 2.0, T_fgr, o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()).transformation,
        }
        # Trimmed ICP from FGR init (numpy)
        try:
            methods["FGR+TrimmedICP"] = trimmed_icp(pts, cadpts, init=T_fgr, threshold=2.0, trim_frac=0.3, max_iter=50)[0]
        except Exception as e:
            print(f"  Trimmed err: {e}")
        scan_rows = []
        for mname, T in methods.items():
            d = p2m(pts @ T[:3, :3].T + T[:3, 3])
            scan_rows.append({"method": mname, "resid_med_mm": float(np.median(d)), "inlier_lt2mm_pct": float(np.mean(d < 2.0) * 100)})
        rows.append({"scan": scan_file, "label": label, "n_pts": len(pts), "results": scan_rows})
        print(f"\n{scan_file} ({label}, {len(pts)} pts) — GeoP2E published: 0.25-0.28 mm")
        for r in scan_rows:
            print(f"  {r['method']:18s} resid_med={r['resid_med_mm']:.3f}mm inlier(<2mm)={r['inlier_lt2mm_pct']:.0f}%")

    out = {"metric": "point-to-CAD-cloud residual (median mm; proxy for point-to-mesh). GeoP2E published = 0.25-0.28 mm (Table 3, cited not re-run).",
           "note": "All baselines use FGR global coarse init (fair: GeoP2E uses hypothesis-verification coarse init). Baselines operate on point clouds (no need for the degenerate coaxial features).",
           "scans": rows}
    os.makedirs("results", exist_ok=True)
    json.dump(out, open("results/gear_baselines.json", "w"), indent=2)
    print("\nSaved results/gear_baselines.json")


if __name__ == "__main__":
    main()
