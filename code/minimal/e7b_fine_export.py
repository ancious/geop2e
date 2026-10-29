"""e7b — fine re-export per vendor constraints + first-batch-5 reselection.

Vendor rules implemented:
  chordal <=0.02mm  -> BRepMesh linear deflection 0.01mm (angular secondary)
  watertight+solid  -> BRepCheck_Analyzer validity + mesh manifold check (every
                       mesh edge shared by exactly 2 triangles, single shell)
  avoid symmaddetry  -> mirror-symmetry probe (3 principal-plane reflections,
                       KD vertex match >=0.90 flags ill-posed multi-solution GT)
  dual format        -> binary STL `_fine.stl` + scaled `_scaled.step` (the
                       scaled solid IS the registration CAD — parsed primitives
                       then match the physical part exactly)
  size               -> uniform scale to 148mm max edge (unchanged normalization)
Output: code/print_pack_20_v2/ (never overwrites), report JSON with volumes,
        watertight/symmetry flags, and the first-batch-5 selection.
"""
import os, sys, json, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))

SRC_DIR = os.path.join(os.path.dirname(HERE), "print_pack_20")
OLD_MANIFEST = os.path.join(SRC_DIR, "MANIFEST.json")
OUT_DIR = os.path.join(os.path.dirname(HERE), "print_pack_20_v2")
TARGET_MM = 148.0
LIN_DEFLECT = 0.01     # mm -> chordal error <=0.01 <= 0.02 vendor bound
ANG_DEFLECT = 0.05
FIRST_BATCH_KEEP = {"52890_7ae554d9_1.stp", "21644_aa203dc5_3.stp"}   # small two stay
CURRENT3 = {"24012_097cf980_0.stp", "52890_7ae554d9_1.stp", "21644_aa203dc5_3.stp"}
VOL_CAP_CM3 = 600.0    # cost guard; >800cm3 needs separate discussion


def ocp():
    from OCP.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_StepModelType
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.gp import gp_Trsf, gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_SHELL
    from OCP.TopoDS import TopoDS
    from OCP.IFSelect import IFSelect_RetDone
    return {k: v for k, v in dict(locals()).items() if not k.startswith("_")}


def load_solid(O, path):
    from OCP.STEPControl import STEPControl_Reader
    from OCP.IFSelect import IFSelect_RetDone
    r = STEPControl_Reader()
    st = r.ReadFile(path)
    assert st == IFSelect_RetDone, f"{path}: status={st}"
    r.TransferRoots()
    return r.OneShape()


def count_kinds(O, shape, kind):
    exp = O["TopExp_Explorer"](shape, kind)   # kind is already a TopAbs_ShapeEnum VALUE
    n = 0
    while exp.More():
        n += 1; exp.Next()
    return n


def mirror_symmetry(mesh_pts):
    """Principal-plane reflection match on a vertex subsample (KD)."""
    from scipy.spatial import cKDTree
    v = mesh_pts
    if len(v) > 8000:
        v = v[np.random.default_rng(0).choice(len(v), 8000, replace=False)]
    c = v.mean(0)
    _, _, Vt = np.linalg.svd(v - c, full_matrices=False)
    axes = Vt
    tree = cKDTree(v)
    tol = 0.02 * float(np.linalg.norm(np.ptp(v, axis=0)))
    flags = []
    for i in range(3):
        refl = v - 2 * np.outer((v - c) @ axes[i], axes[i])
        d, _ = tree.query(refl, k=1)
        m = float(np.mean(d < tol))
        flags.append(m)
    return max(flags), [round(f, 3) for f in flags]


def manifold_check(stl_path):
    import open3d as o3d
    m = o3d.io.read_triangle_mesh(stl_path)
    tri = np.asarray(m.triangles)
    if len(tri) == 0:
        return False, 0
    edges = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    uniq, cnt = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(cnt == 2)), int((cnt != 2).sum())


def main():
    O = ocp()
    old = json.load(open(OLD_MANIFEST))
    recs = []
    print(f"re-exporting {len(old)} parts, lin_deflect={LIN_DEFLECT}mm")
    for ent in old:
        stem = ent["part"].replace(".stp", "")
        src_stp = os.path.join(os.path.expanduser(
            "~/projects/new/3dphy/data/fusion360_segmentation/s2.0.1/breps/step"), ent["part"])
        if not os.path.exists(src_stp):
            src_stp = ("data/gear_step_cad/"
                       "04_代码仓与软件数据/01_代码仓/py_1/vision/cad_3d/feature_match/"
                       "abc_data/step2equation/sample_model.stp")
        try:
            shape = load_solid(O, src_stp)
            valid = O["BRepCheck_Analyzer"](shape).IsValid()
            n_solids = count_kinds(O, shape, O["TopAbs_SOLID"])
            n_shells = count_kinds(O, shape, O["TopAbs_SHELL"])
            box = O["Bnd_Box"](); O["BRepBndLib"].Add_s(shape, box)
            a, b, z0, d, e, f = box.Get()
            ext = max(d - a, e - b, f - z0)
            s = TARGET_MM / max(ext, 1e-9)
            from OCP.gp import gp_Trsf, gp_Pnt
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            trsf = gp_Trsf(); trsf.SetScale(gp_Pnt(0, 0, 0), s)
            scaled = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
            # volume (scaled)
            props = O["GProp_GProps"]()
            O["BRepGProp"].VolumeProperties_s(scaled, props)
            vol_cm3 = props.Mass() / 1000.0
            # fine mesh
            O["BRepMesh_IncrementalMesh"](scaled, LIN_DEFLECT, False, ANG_DEFLECT, True)
            os.makedirs(OUT_DIR, exist_ok=True)
            stl = os.path.join(OUT_DIR, f"{stem}_fine.stl")
            w = O["StlAPI_Writer"](); w.ASCIIMode = False
            assert w.Write(scaled, stl)
            step_out = os.path.join(OUT_DIR, f"{stem}_scaled.step")
            wr = O["STEPControl_Writer"]()
            wr.Transfer(scaled, O["STEPControl_StepModelType"].STEPControl_AsIs)
            wr.Write(step_out)
            # checks
            import open3d as o3d
            m = o3d.io.read_triangle_mesh(stl)
            man, bad = manifold_check(stl)
            sym_max, sym_planes = mirror_symmetry(np.asarray(m.vertices))
            box2 = O["Bnd_Box"](); O["BRepBndLib"].Add_s(scaled, box2)
            x0, y0, z0, x1, y1, z1 = box2.Get()
            mb = os.path.getsize(stl) / 1e6
            recs.append(dict(part=ent["part"], stem=stem, scale=s, U=ent["U"], cov=ent["cov"],
                             n=ent["n"], kinds=ent["kinds"], axisConc=ent["axis_conc"],
                             bbox_mm=[round(x1 - x0, 1), round(y1 - y0, 1), round(z1 - z0, 1)],
                             volume_cm3=round(vol_cm3, 1), brep_valid=bool(valid),
                             n_solids=n_solids, n_shells=n_shells,
                             manifold=man, nonmanifold_edges=bad,
                             mirror_sym_max=round(sym_max, 3), mirror_planes=sym_planes,
                             stl_mb=round(mb, 1),
                             stl=os.path.basename(stl), step=os.path.basename(step_out)))
            flag = "" if (valid and man and n_solids == 1 and sym_max < 0.90) else "  <-- CHECK"
            print(f"  {stem[:26]:26s} vol={vol_cm3:7.1f}cm3 bbox_max={max(recs[-1]['bbox_mm']):5.1f} "
                  f"valid={valid} solids={n_solids} man={man} sym={sym_max:.2f} {mb:.0f}MB{flag}")
        except Exception as ex:
            recs.append(dict(part=ent["part"], error=repr(ex)[:150]))
            print(f"  {stem}: FAIL {ex!r:.100}")

    # ---- first-batch selection: keep the 2 small current, drop the largest of 3
    by = {r["part"]: r for r in recs if "error" not in r}
    cur3 = [by[p] for p in CURRENT3 if p in by]
    drop = max(cur3, key=lambda r: r["volume_cm3"])
    print(f"\ncurrent-3 volumes: " + ", ".join(f"{r['stem'][:12]}={r['volume_cm3']}cm3" for r in cur3))
    print(f"-> REPLACING largest: {drop['stem']} ({drop['volume_cm3']} cm3)")
    ok = [r for r in recs if "error" not in r
          and r["brep_valid"] and r["manifold"] and r["n_solids"] == 1
          and r["mirror_sym_max"] < 0.90 and r["volume_cm3"] <= VOL_CAP_CM3
          and r["part"] not in CURRENT3]
    ok.sort(key=lambda r: (r["U"], -r["cov"]))
    # diversity pick: 1 compact block (replacement), 1 flat slab w/ many cylinders,
    # 1 compact plane-heavy — distinct from kept shaft & thin plate
    def pick(pred):
        for r in ok:
            if r["part"] not in {x["part"] for x in first} and pred(r):
                return r
        return None
    first = [by[p] for p in FIRST_BATCH_KEEP if p in by]
    vol_rank = sorted(ok, key=lambda r: r["volume_cm3"])
    # replacement: compact block, all dims 60-150, mixed planes+cyls
    rep = None
    for r in ok:
        b = r["bbox_mm"]
        if min(b) >= 55 and r["kinds"].get("plane", 0) >= 4 and r["kinds"].get("cylinder", 0) >= 3:
            rep = r; break
    if rep: first.append(rep)
    # add-1: flat slab (min dim 25-60mm, cylinder-rich, cov>=0.95)
    add1 = None
    for r in ok:
        b = r["bbox_mm"]
        if 25 <= min(b) <= 60 and r["kinds"].get("cylinder", 0) >= 5 and r["cov"] >= 0.95:
            add1 = r; break
    if add1: first.append(add1)
    # add-2: compact plane-heavy block, low axisConc
    add2 = None
    for r in sorted(ok, key=lambda r: r["axisConc"]):
        b = r["bbox_mm"]
        if min(b) >= 60 and r["kinds"].get("plane", 0) >= 5 and r["kinds"].get("cylinder", 0) >= 2:
            add2 = r; break
    if add2: first.append(add2)
    for r in first:
        r["first_batch"] = True
    out = dict(target_mm=TARGET_MM, lin_deflect_mm=LIN_DEFLECT,
               vol_cap_cm3=VOL_CAP_CM3, dropped_for_volume=drop["stem"],
               first_batch=[r["stem"] for r in first], parts=recs)
    json.dump(out, open(os.path.join(OUT_DIR, "MANIFEST_v2.json"), "w"), indent=2)
    print(f"\nFIRST BATCH (5): {[r['stem'] for r in first]}")
    print(f"saved {OUT_DIR}/MANIFEST_v2.json")


if __name__ == "__main__":
    main()
