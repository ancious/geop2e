"""e7_print_select — pick 20 CAD parts GeoP2E can register accurately, scale to a
15 cm cube, export print-ready STLs.

Selection criteria (derived from the diagnosed failure modes, not aesthetics):
  1. parsed analytic coverage >= 0.8 (plane/cyl/sphere B-Rep support)
  2. >= 6 merged supports with >=2 distinct PRIMITIVE KINDS
  3. normalized-FIM uncertainty U = max(R95/5deg, T95/2%D) <= 1 on ALL supports
     (the A3 alarm rule: quantifies actual registrability, incl. yaw modes)
  4. anti-symmetry: penalise parts whose cylinder support-area concentrates on a
     single common axis (>70% -> yaw-ambiguous, the S1 failure)
  5. size sanity after scaling (bbox fits 150mm cube by uniform scale)
Ranking score: U ascending (lower = better conditioned), tiebreak coverage desc,
support count desc, kind diversity desc. Top 20 exported.
"""
import os, sys, json, glob, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import e6_auto as A

STEP_DIR = os.path.expanduser("~/projects/new/3dphy/data/fusion360_segmentation/subset100/s2.0.1/breps/step")
ABC_STEP = "data/gear_step_cad/sample_model.stp"
OUT_DIR = os.path.join(HERE, "..", "print_pack_20")
TARGET_MM = 148.0   # discretisation bulge margin -> guaranteed <=150mm


def part_U(part, sigma=None):
    """Normalized-FIM uncertainty U over ALL supports (A3 formula)."""
    sup = part["supports"]
    guides = [m["pts"] for m in sup]
    D = part["D"]
    k_H = 0.012 * D
    sigma = sigma or 0.2 * k_H
    _, table = A.select_auto(part, sigma=sigma, val_score_fn=lambda idx: (0.0, 0.0))
    allrow = next((r for r in table if r["name"] == "all"), None)
    return allrow["U"] if allrow else np.inf


def axis_concentration(part):
    """Share of cylinder SUPPORT POINTS on the dominant cylinder axis (yaw risk)."""
    cyl = [m for m in part["supports"] if m["feat"].kind == "cylinder"]
    n_all = sum(len(m["pts"]) for m in part["supports"]) or 1
    if not cyl:
        return 0.0
    axes = [m["feat"].axis / np.linalg.norm(m["feat"].axis) for m in cyl]
    counts = [len(m["pts"]) for m in cyl]
    best = 0
    for i, a in enumerate(axes):
        same = sum(c for j, c in enumerate(counts)
                   if abs(abs(float(axes[j] @ a)) - 1) < 1e-3)
        best = max(best, same)
    return best / n_all


def scale_export_stl(step_path, out_stl):
    """Uniform-scale the STEP solid to fit TARGET_MM cube; write binary STL (mm)."""
    from OCP.STEPControl import STEPControl_Reader
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.StlAPI import StlAPI_Writer
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.gp import gp_Trsf, gp_Vec, gp_Pnt
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.TopoDS import TopoDS_Shape
    from OCP.IFSelect import IFSelect_RetDone

    r = STEPControl_Reader()
    assert r.ReadFile(step_path) == IFSelect_RetDone
    r.TransferRoots()
    shape = r.OneShape()
    box = Bnd_Box()
    BRepBndLib.Add_s(shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    ext = max(xmax - xmin, ymax - ymin, zmax - zmin)
    s = TARGET_MM / ext
    trsf = gp_Trsf(); trsf.SetScale(gp_Pnt(0, 0, 0), s)
    scaled = BRepBuilderAPI_Transform(shape, trsf, True).Shape()
    BRepMesh_IncrementalMesh(scaled, 0.5, False, 0.5, True)
    w = StlAPI_Writer()
    w.ASCIIMode = False
    assert w.Write(scaled, out_stl)
    box2 = Bnd_Box(); BRepBndLib.Add_s(scaled, box2)
    a, b, c, d, e, f = box2.Get()
    return s, (d - a, e - b, f - c)


def main():
    paths = sorted(glob.glob(os.path.join(STEP_DIR, "*.stp"))) + [ABC_STEP]
    print(f"screening {len(paths)} STEP parts")
    cands = []
    for p in paths:
        try:
            part = A.load_parsed_part(p)
        except Exception as ex:
            continue
        if part is None or part["analytic_coverage"] < 0.8 or len(part["supports"]) < 6:
            continue
        kinds = {}
        for m in part["supports"]:
            kinds[m["feat"].kind] = kinds.get(m["feat"].kind, 0) + 1
        if len(kinds) < 2:
            continue
        U = part_U(part)
        ax = axis_concentration(part)
        cands.append(dict(part=os.path.basename(p), path=p, U=round(U, 4),
                          cov=part["analytic_coverage"], n=len(part["supports"]),
                          kinds=kinds, axis_conc=round(ax, 3),
                          score=(round(U, 3), -part["analytic_coverage"], -len(part["supports"]))))
        print(f"  {os.path.basename(p)[:28]:28s} U={U:.3f} cov={part['analytic_coverage']:.2f} "
              f"n={len(part['supports']):3d} {kinds} axisConc={ax:.2f}")
    # eligibility: U<=1, axis_conc<0.70
    ok = [c for c in cands if c["U"] <= 1.0 and c["axis_conc"] < 0.70]
    ok.sort(key=lambda c: c["score"])
    chosen = ok[:20]
    if len(chosen) < 20:            # backfill from near-miss (elevated axis conc), flagged
        near = sorted([c for c in cands if c["U"] <= 1.0 and c["axis_conc"] >= 0.70],
                      key=lambda c: c["score"])
        for c in near[: 20 - len(chosen)]:
            c["note"] = "backfill: elevated single-axis cylinder concentration (yaw-symmetry risk)"
            chosen.append(c)
    print(f"\neligible={len(ok)}; exporting {len(chosen)} STLs to {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)
    recs = []
    for c in chosen:
        out_stl = os.path.join(OUT_DIR, c["part"].replace(".stp", "") + "_150mm.stl")
        try:
            s, bbox = scale_export_stl(c["path"], out_stl)
            recs.append(dict(**{k: c[k] for k in ("part", "U", "cov", "n", "kinds", "axis_conc")},
                             stl=os.path.basename(out_stl), scale=s,
                             bbox_mm=[round(x, 1) for x in bbox]))
            print(f"  exported {recs[-1]['stl']} bbox={recs[-1]['bbox_mm']} (x{s:.3f})")
        except Exception as ex:
            print(f"  EXPORT FAIL {c['part']}: {ex!r:.100}")
    json.dump(recs, open(os.path.join(OUT_DIR, "MANIFEST.json"), "w"), indent=2)
    print(f"saved {OUT_DIR}/MANIFEST.json with {len(recs)} parts")


if __name__ == "__main__":
    main()
