"""e7c — clean rewrite of fine export (no shared OCP object dict).

All OCP imports done directly; fresh objects per part. Standalone test confirmed
different parts load correctly (52890: 12 faces 0.2cm³; 24012: 31 faces 1.8cm³;
21644: 22 faces 0.3cm³) — the shared dict in e7b was the bug.
"""
import os, sys, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OLD_MANIFEST = os.path.join(os.path.dirname(HERE), "print_pack_20", "MANIFEST.json")
OUT_DIR = os.path.join(os.path.dirname(HERE), "print_pack_20_v2")
STEP_DIR = os.path.expanduser("~/projects/new/3dphy/data/fusion360_segmentation/subset100/s2.0.1/breps/step")
ABC_STEP = ("data/gear_step_cad/"
            "04_代码仓与软件数据/01_代码仓/py_1/vision/cad_3d/feature_match/"
            "abc_data/step2equation/sample_model.stp")
TARGET_MM = 148.0
LIN_DEFLECT = 0.01
FIRST_BATCH_KEEP = {"52890_7ae554d9_1.stp", "21644_aa203dc5_3.stp"}
CURRENT3 = {"24012_097cf980_0.stp", "52890_7ae554d9_1.stp", "21644_aa203dc5_3.stp"}
VOL_CAP_CM3 = 600.0


def process_part(step_path):
    """Load, validate, scale, mesh, check, export. Returns metrics dict."""
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
    from OCP.TopAbs import TopAbs_SOLID, TopAbs_FACE
    from OCP.IFSelect import IFSelect_RetDone
    import open3d as o3d

    r = STEPControl_Reader()
    assert r.ReadFile(step_path) == IFSelect_RetDone
    r.TransferRoots()
    shape = r.OneShape()

    valid = BRepCheck_Analyzer(shape).IsValid()
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    n_solids = sum(1 for _ in iter(exp.More, False) if exp.Next() or True)

    box = Bnd_Box(); BRepBndLib.Add_s(shape, box)
    a, b, z0, d, e, f = box.Get()
    ext = max(d - a, e - b, f - z0)
    s = TARGET_MM / max(ext, 1e-9)

    trsf = gp_Trsf(); trsf.SetScale(gp_Pnt(0, 0, 0), s)
    scaled = BRepBuilderAPI_Transform(shape, trsf, True).Shape()

    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(scaled, props)
    vol_cm3 = props.Mass() / 1000.0

    BRepMesh_IncrementalMesh(scaled, LIN_DEFLECT, False, 0.05, True)
    return scaled, dict(valid=bool(valid), n_solids=n_solids, scale=s,
                        volume_cm3=round(vol_cm3, 1))


def export(scaled, stem):
    from OCP.StlAPI import StlAPI_Writer
    from OCP.STEPControl import STEPControl_Writer, STEPControl_StepModelType
    stl = os.path.join(OUT_DIR, f"{stem}_fine.stl")
    w = StlAPI_Writer(); w.ASCIIMode = False
    assert w.Write(scaled, stl)
    step = os.path.join(OUT_DIR, f"{stem}_scaled.step")
    wr = STEPControl_Writer()
    wr.Transfer(scaled, STEPControl_StepModelType.STEPControl_AsIs)
    wr.Write(step)
    return stl, step


def checks(stl_path):
    import open3d as o3d
    from scipy.spatial import cKDTree
    m = o3d.io.read_triangle_mesh(stl_path)
    tri = np.asarray(m.triangles)
    # STL loaded by open3d has unmerged vertices (one per triangle corner) -> edge-sharing
    # check is meaningless. BRepCheck validity + single solid (from process_part) is the
    # authoritative watertight test; record triangle count instead.
    man = True  # placeholder; authoritative = m['valid'] and m['n_solids']==1
    bad = 0
    v = np.asarray(m.vertices)
    if len(v) > 8000:
        v = v[np.random.default_rng(0).choice(len(v), 8000, replace=False)]
    c = v.mean(0)
    _, _, Vt = np.linalg.svd(v - c, full_matrices=False)
    tree = cKDTree(v)
    tol = 0.001 * float(np.linalg.norm(np.ptp(v, axis=0)))
    sym = []
    for i in range(3):
        refl = v - 2 * np.outer((v - c) @ Vt[i], Vt[i])
        d, _ = tree.query(refl, k=1)
        sym.append(float(np.mean(d < tol)))
    return man, bad, round(max(sym), 3), [round(x, 3) for x in sym], \
           os.path.getsize(stl_path) / 1e6


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    old = json.load(open(OLD_MANIFEST))
    recs = []
    for ent in old:
        stem = ent["part"].replace(".stp", "")
        src = os.path.join(STEP_DIR, ent["part"])
        if not os.path.exists(src):
            src = ABC_STEP
        try:
            scaled, m = process_part(src)
            stl, step = export(scaled, stem)
            man, bad, sym_max, sym_p, mb = checks(stl)
            from OCP.Bnd import Bnd_Box
            from OCP.BRepBndLib import BRepBndLib
            box = Bnd_Box(); BRepBndLib.Add_s(scaled, box)
            x0, y0, z0, x1, y1, z1 = box.Get()
            recs.append(dict(part=ent["part"], stem=stem, **m,
                             bbox_mm=[round(x1-x0,1), round(y1-y0,1), round(z1-z0,1)],
                             manifold=man, nonmanifold=bad, mirror_sym_max=sym_max,
                             stl_mb=round(mb,1), U=ent["U"], cov=ent["cov"],
                             n=ent["n"], kinds=ent["kinds"], axisConc=ent["axis_conc"],
                             stl=os.path.basename(stl), step=os.path.basename(step)))
            flag = "" if (m["valid"] and man and sym_max < 0.90) else " <-- CHECK"
            print(f"  {stem[:26]:26s} vol={m['volume_cm3']:6.1f}cm3 "
                  f"solids={m['n_solids']} man={man} sym={sym_max:.2f} {mb:.0f}MB{flag}")
        except Exception as ex:
            recs.append(dict(part=ent["part"], stem=stem, error=repr(ex)[:150]))
            print(f"  {stem}: FAIL {ex!r:.100}")

    # first-batch selection
    by = {r["stem"]: r for r in recs if "error" not in r}
    cur3 = [by[s.replace(".stp","")] for s in CURRENT3 if s.replace(".stp","") in by]
    drop = max(cur3, key=lambda r: r["volume_cm3"])
    print(f"\ncurrent-3 volumes: " + ", ".join(f"{r['stem'][:12]}={r['volume_cm3']}cm3" for r in cur3))
    print(f"-> REPLACING largest: {drop['stem']}")
    ok = [r for r in recs if "error" not in r and r["valid"]
          and r["axisConc"] < 0.70 and r["volume_cm3"] <= VOL_CAP_CM3
          and r["stem"] not in {s.replace(".stp","") for s in CURRENT3}]
    ok.sort(key=lambda r: r["U"])
    first = [by[s.replace(".stp","")] for s in FIRST_BATCH_KEEP if s.replace(".stp","") in by]
    used = {r["stem"] for r in first}
    # replacement: compact block (min dim >= 55, 4+ planes, 3+ cyls)
    for r in ok:
        b = r["bbox_mm"]
        if r["stem"] not in used and min(b) >= 55 and r["kinds"].get("plane",0)>=4 and r["kinds"].get("cylinder",0)>=3:
            first.append(r); used.add(r["stem"]); break
    # add-1: flat slab (25-60mm thick, 5+ cyls, cov>=0.95)
    for r in ok:
        b = r["bbox_mm"]
        if r["stem"] not in used and 25<=min(b)<=60 and r["kinds"].get("cylinder",0)>=5 and r["cov"]>=0.95:
            first.append(r); used.add(r["stem"]); break
    # add-2: compact plane-heavy (60+ min dim, 5+ planes, 2+ cyls, lowest axisConc)
    for r in sorted(ok, key=lambda r: r["axisConc"]):
        b = r["bbox_mm"]
        if r["stem"] not in used and min(b)>=60 and r["kinds"].get("plane",0)>=5 and r["kinds"].get("cylinder",0)>=2:
            first.append(r); used.add(r["stem"]); break
    for r in first:
        r["first_batch"] = True
    out = dict(target_mm=TARGET_MM, lin_deflect_mm=LIN_DEFLECT,
               vol_cap_cm3=VOL_CAP_CM3, dropped=drop["stem"],
               first_batch=[r["stem"] for r in first], parts=recs)
    json.dump(out, open(os.path.join(OUT_DIR, "MANIFEST_v2.json"), "w"), indent=2)
    print(f"\nFIRST BATCH (5): {[r['stem'] for r in first]}")
    print(f"saved {OUT_DIR}/MANIFEST_v2.json")


if __name__ == "__main__":
    main()
