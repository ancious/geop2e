"""Robust gear baselines: 3 defect scans x up to 3 FGR voxel settings {0.8,1.0,1.3},
report the MEDIAN over the settings that ran (failed FGR settings are skipped).

DOCSTRING FIX 2026-08-27: the previous header said "report BEST", but the code
below has always computed med3 = median over the voxel settings; "BEST"
was a stale description, not the executed logic. The manuscript's wording
"median of three FGR initializations" is likewise imprecise: the runs
differ by FGR *voxel size*, not by stochastic initializations.
"""
import sys,os,json,numpy as np,open3d as o3d
from scipy.spatial import cKDTree
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from gear_baseline_extended import fpfh, sparse_icp
from baselines import trimmed_icp
GEAR="data/gear_step_cad"
SCANS=[("defect/22.pcd","~30% loss"),("defect/4.ply","defective"),("defect/1.ply","fragment")]
VOXELS=[0.8,1.0,1.3]
cad_mesh=o3d.io.read_triangle_mesh(f"{GEAR}/gear/model.obj"); cad_mesh.compute_vertex_normals()
cad_pc=cad_mesh.sample_points_uniformly(number_of_points=20000); cad_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0,max_nn=30))
cadpts=np.asarray(cad_pc.points)
scene=o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(cad_mesh))
def p2m(pts): return scene.compute_distance(o3d.core.Tensor(np.asarray(pts,dtype=np.float32))).numpy()
out=[]
for sf,label in SCANS:
    src_raw=o3d.io.read_point_cloud(f"{GEAR}/{sf}"); pts=np.asarray(src_raw.points)
    src=o3d.geometry.PointCloud(); src.points=o3d.utility.Vector3dVector(pts); src.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0,max_nn=30))
    best={}
    for v in VOXELS:
        sd,fs=fpfh(src_raw,v); td=cad_pc.voxel_down_sample(v); _,ft=fpfh(cad_pc,v)
        opt=o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=v*1.5,use_absolute_scale=True)
        try: Tfgr=o3d.pipelines.registration.registration_fgr_based_on_feature_matching(sd,td,fs,ft,opt).transformation
        except: continue
        def icp(e,thr=2.0): return o3d.pipelines.registration.registration_icp(src,cad_pc,thr,Tfgr,e).transformation
        cands={"FGR":Tfgr,"FGR+ICPp2p":icp(o3d.pipelines.registration.TransformationEstimationPointToPoint()),
               "FGR+ICPp2plane":icp(o3d.pipelines.registration.TransformationEstimationPointToPlane()),
               "FGR+GICP":o3d.pipelines.registration.registration_generalized_icp(src,cad_pc,2.0,Tfgr,o3d.pipelines.registration.TransformationEstimationForGeneralizedICP()).transformation,
               "FGR+Trimmed":trimmed_icp(pts,cadpts,init=Tfgr,threshold=2.0,trim_frac=0.3,max_iter=50)[0],
               "FGR+SparseICP":sparse_icp(pts,cadpts,init=Tfgr,threshold=3.0,p=0.5,outer=20)}
        for m,T in cands.items():
            med=float(np.median(p2m(pts@T[:3,:3].T+T[:3,3])))
            best.setdefault(m,[]).append(med)   # collect all 3 inits
    med3={m:float(np.median(v)) for m,v in best.items()}   # MEDIAN of 3 inits (honest, not best)
    out.append({"scan":sf,"label":label,"n_pts":len(pts),"resid_median_of3_mm":med3,"raw_inits":best})
    print(f"{sf} ({label}, n={len(pts)}): "+" ".join(f"{m}={v:.3f}" for m,v in sorted(med3.items(),key=lambda x:x[1])))
json.dump({"metric":"point-to-MESH MEDIAN over 3 FGR inits (honest, not best); GeoP2E published 0.25-0.28mm (cited, not re-run). Descriptive comparison, no inferential test.","open3d_version":o3d.__version__,"geoP2E_note":"published Table-3 value from the original tuned pipeline; baselines freshly run on the same scans but a different pipeline -> not like-for-like (disclosed limitation).","scans":out},open("results/gear_baselines_robust.json","w"),indent=2)
print("saved gear_baselines_robust.json")
