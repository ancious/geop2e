"""e4p — E4': parsed-primitive (STEP->B-Rep) non-gear study. Replaces the voided
fitted-primitive E4 (operator constraint: fitted primitives NOT acceptable).

Protocol (GPT-prescribed):
- screen ALL 118 local STEP parts; eligibility: >=5 merged plane/cyl/sphere
  supports AND analytic coverage >=0.80 (primary) / 0.50-0.80 (secondary stratum);
  ineligible recorded with reason. No 'industrial-looking' cherry-picking.
- design groups (filename design id) -> FROZEN 25% dev / 75% test split manifest
  (sha256-hashed) BEFORE any trial; test groups untouched by tuning.
- per test part: 18 main trials = retained fraction {1.0,0.8,0.6} x noise
  sigma/D {5e-4,2e-3} x 3 transforms (rot 5-45deg, trans +-0.1D, half-space crop
  along random axis at the retention quantile, sensor-visibility normal test)
  + 3 clutter trials (keep 0.8, sigma 1e-3 D, +10% uniform bbox outliers).
- arms (same inputs, shared per-trial FGR start): FGR / FGR+ICP_p2plane /
  GeoP2E-production (FIM-8, production solver) / GeoP2E-auto (A3 selection ONCE
  per part; A4 n_starts=3; staged HV max_cands=4 — compute-budget reduction,
  declared). All arms see the SAME support cloud (3-type faces only), disclosed.
- metrics: raw RRE/RTE vs known GT, success = RRE<5deg & RTE<2%D (predeclared),
  trimmed-residual score, runtime; paired per trial; part-clustered bootstrap CIs.
Usage: python e4p_parsed_study.py [--limit N] [--wave main] [--out F]
"""
import os, sys, json, time, glob, hashlib, argparse
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rot

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import e0_common as e0
import e6_auto as A
from solver import estimate_normals, register
from selection import greedy_select

STEP_DIR = os.path.expanduser("~/projects/new/3dphy/data/fusion360_segmentation/subset100/s2.0.1/breps/step")
MANIFEST = os.path.join(HERE, "artifacts", "e4p_split_manifest.json")
SPLIT_SEED = 20260827
SCAN_N = 15000          # scan subsample cap (compute budget, declared)
N_TR_PER_CELL = 3


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def build_manifest(paths, limit=None):
    """Parse all parts, apply eligibility, group by design id, freeze dev/test."""
    if os.path.exists(MANIFEST):
        return json.load(open(MANIFEST))
    recs, ineligible = [], []
    for p in sorted(paths)[: (limit or 10**9)]:
        try:
            part = A.load_parsed_part(p)
        except Exception as ex:
            ineligible.append(dict(part=os.path.basename(p), reason=f"parse:{ex!r:.80}"))
            continue
        if part is None:
            ineligible.append(dict(part=os.path.basename(p), reason="<3 supports"))
            continue
        cov, ns = part["analytic_coverage"], len(part["supports"])
        if ns < 5:
            ineligible.append(dict(part=os.path.basename(p), reason=f"supports={ns}<5"))
            continue
        stratum = "primary" if cov >= 0.80 else ("secondary" if cov >= 0.50 else None)
        if stratum is None:
            ineligible.append(dict(part=os.path.basename(p), reason=f"coverage={cov:.2f}<0.50"))
            continue
        design = "_".join(os.path.basename(p).split("_")[:2])
        recs.append(dict(part=os.path.basename(p), path=p, sha16=sha(p), design=design,
                         supports=ns, coverage=round(cov, 3), stratum=stratum))
    groups = sorted({r["design"] for r in recs})
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(groups)
    n_dev = max(1, int(round(0.25 * len(groups))))
    dev_groups = set(groups[:n_dev])
    for r in recs:
        r["split"] = "dev" if r["design"] in dev_groups else "test"
    m = dict(seed=SPLIT_SEED, n_groups=len(groups), n_dev=n_dev,
             parts=recs, ineligible=ineligible,
             note="FROZEN before any trial; dev used only for mechanism calibration")
    json.dump(m, open(MANIFEST, "w"), indent=2)
    print(f"manifest frozen: {len(recs)} eligible ({sum(1 for r in recs if r['split']=='test')} test), "
          f"{len(ineligible)} ineligible")
    return m


def synth_scan(tcad, rng, keep, sigma_frac, outliers=0.0, rot_deg=None, trans_frac=None):
    pts, nrm = tcad.part["_scan_pts"], tcad.part["_scan_nrms"]
    v = rng.normal(size=3); v /= np.linalg.norm(v)
    vis = (nrm @ v) < 0.1                       # sensor-visibility (facing) test
    if keep < 1.0:
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        proj = pts @ ax
        cut = np.quantile(proj, keep)
        side = rng.random() < 0.5
        keep_m = (proj > cut) if side else (proj < cut)
        keep_m &= vis
    else:
        keep_m = vis | (rng.random(len(pts)) < 0.3)   # keep some backside at keep=1
    src = pts[keep_m]
    D = tcad.D
    rot_deg = rng.uniform(5, 45) if rot_deg is None else rot_deg
    trans_frac = rng.uniform(0.02, 0.10) if trans_frac is None else trans_frac
    axr = rng.normal(size=3); axr /= np.linalg.norm(axr)
    R = Rot.from_rotvec(axr * np.deg2rad(rot_deg)).as_matrix()
    t = rng.uniform(-1, 1, 3) * D * trans_frac
    src = src @ R.T + t + rng.normal(0, sigma_frac * D, src.shape)
    if outliers > 0:
        lo, hi = src.min(0), src.max(0)
        n_out = int(outliers * len(src))
        src = np.vstack([src, rng.uniform(lo, hi, (n_out, 3))])
    T_gt = np.eye(4); T_gt[:3, :3] = R; T_gt[:3, 3] = t     # cad -> src
    if len(src) > SCAN_N:
        idx = rng.choice(len(src), SCAN_N, replace=False)
        src = src[idx]
    return src, T_gt


def run_part(part_rec, tcad, sel_idx, rng_seed_base, out_rows):
    """All trial bodies are fault-isolated: an exception (or a degenerate scan)
    records an error row and CONTINUES — one bad viewpoint must not kill the
    735-trial batch."""
    D = tcad.D
    k_H = 0.012 * D
    sub_tcad = A.TrimmedCAD_from_subset(tcad, sel_idx) if sel_idx else tcad
    cells = [dict(keep=k, sig=s) for k in (1.0, 0.8, 0.6) for s in (5e-4, 2e-3)] + \
            [dict(keep=0.8, sig=1e-3, clutter=0.10)]
    for ci, cell in enumerate(cells):
        for tr in range(N_TR_PER_CELL):
            seed = rng_seed_base + ci * 100 + tr
            try:
                _run_trial(part_rec, tcad, sub_tcad, seed, ci, cell, tr, D, k_H, out_rows)
            except Exception as ex:
                out_rows.append(dict(part=part_rec["part"], design=part_rec["design"],
                                     cell=ci, trial=tr, seed=seed,
                                     error=repr(ex)[:200]))
                print(f"  {part_rec['part'][:18]:18s} c{ci} t{tr}: TRIAL-ERROR {ex!r:.100}",
                      flush=True)


def _run_trial(part_rec, tcad, sub_tcad, seed, ci, cell, tr, D, k_H, out_rows):
    rng = np.random.default_rng(seed)
    src, T_gt = synth_scan(tcad, rng, cell["keep"], cell["sig"],
                           outliers=cell.get("clutter", 0.0))
    if len(src) < 200:
        raise RuntimeError(f"degenerate scan ({len(src)} pts after visibility crop)")
    sn = estimate_normals(src, radius=0.02 * D, max_nn=30)
    src_pc = o3d.geometry.PointCloud(); src_pc.points = o3d.utility.Vector3dVector(src)
    src_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * D, max_nn=30))
    tgt_pc = o3d.geometry.PointCloud(); tgt_pc.points = o3d.utility.Vector3dVector(tcad.pts)
    tgt_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * D, max_nn=30))
    # shared FGR start (re-estimate normals after voxel downsample - o3d can drop them)
    sd = src_pc.voxel_down_sample(0.01 * D)
    td = tgt_pc.voxel_down_sample(0.01 * D)
    for p_ in (sd, td):
        p_.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.02 * D, max_nn=30))
    fs = o3d.pipelines.registration.compute_fpfh_feature(sd, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05 * D, max_nn=100))
    ft = o3d.pipelines.registration.compute_fpfh_feature(td, o3d.geometry.KDTreeSearchParamHybrid(radius=0.05 * D, max_nn=100))
    opt = o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=0.02 * D, use_absolute_scale=True)
    t0 = time.perf_counter()
    try:
        T_fgr = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(sd, td, fs, ft, opt).transformation
        ok_fgr = True
    except Exception:
        T_fgr = np.eye(4); ok_fgr = False
    t_fgr = time.perf_counter() - t0
    row = dict(part=part_rec["part"], design=part_rec["design"], cell=ci,
               keep=cell["keep"], sigma=cell["sig"], clutter=cell.get("clutter", 0),
               trial=tr, seed=seed, n_src=len(src))
    from solver import evaluate_alignment

    def add(name, T, t):
        re, te = evaluate_alignment(T, T_gt)
        p = src @ T[:3, :3].T + T[:3, 3]
        d, _ = tcad.d_and_label(p)
        row[name] = dict(rre_deg=float(re), rte_frac=float(te / D),
                         success=bool(re < 5.0 and te < 0.02 * D),
                         score=A.robust_score(d, k_H), runtime_s=t)

    add("FGR", T_fgr, t_fgr)
    t0 = time.perf_counter()
    T_icp = o3d.pipelines.registration.registration_icp(
        src_pc, tgt_pc, 0.05 * D, T_fgr,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=100)).transformation
    add("FGR+ICP", T_icp, t_fgr + time.perf_counter() - t0)
    # production GeoP2E: FIM-8 on sub_tcad, production solver
    guides = [m["pts"][np.random.default_rng(1).choice(len(m["pts"]), min(80, len(m["pts"])), replace=False)]
              for m in sub_tcad.part["supports"]]
    sel = greedy_select(sub_tcad.feats, guides, criterion="e", target_size=8,
                        min_features=5, rng=np.random.default_rng(0))
    REG = dict(init_radius=0.18 * D, anneal=0.9, max_iter=60, lam=1e-3,
               k_huber=k_H, angle_thresh_deg=30.0)
    t0 = time.perf_counter()
    try:
        T_prod, _ = register(src, sn, [sub_tcad.feats[i] for i in sel],
                             init_T=T_fgr.copy(), **REG)
        add("GeoP2E_prod", T_prod, t_fgr + time.perf_counter() - t0)
    except Exception as ex:
        row["GeoP2E_prod"] = {"error": repr(ex)[:120]}
    # GeoP2E-auto (A3 done once per part; here A4(3 starts) + HV(4 cands))
    t0 = time.perf_counter()
    try:
        T_ms, _ = A.multistart_auto(tcad, src, sn, k_H, T_fgr.copy(), n_starts=3)
        T_hv, _ = A.staged_hv_auto(tcad, src, sn, k_H, T_ms, max_cands=4)
        add("GeoP2E_auto", T_hv, time.perf_counter() - t0)
    except Exception as ex:
        row["GeoP2E_auto"] = {"error": repr(ex)[:120]}
    out_rows.append(row)
    g = row.get("GeoP2E_auto", {}); pr = row.get("GeoP2E_prod", {}); ic = row["FGR+ICP"]
    print(f"  {part_rec['part'][:18]:18s} c{ci} t{tr}: ICP {'ok' if ic['success'] else 'X'} "
          f"({ic['rre_deg']:.1f}) | prod {'ok' if pr.get('success') else 'X'} "
          f"({pr.get('rre_deg', -1):.1f}) | auto {'ok' if g.get('success') else 'X'} "
          f"({g.get('rre_deg', -1):.1f})", flush=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "e4p_parsed_study.json"))
    args = ap.parse_args()
    paths = sorted(glob.glob(os.path.join(STEP_DIR, "*.stp")))
    m = build_manifest(paths, args.limit)
    test_parts = [r for r in m["parts"] if r["split"] == args.split]
    print(f"{len(test_parts)} {args.split} parts")
    rows = []
    done_parts = set()
    ckpt = args.out + ".ckpt"
    if os.path.exists(ckpt):
        import json as _j0
        try:
            rows = _j0.load(open(ckpt))
            from collections import Counter as _C
            cnt = _C(r["part"] for r in rows)
            done_parts = {p for p, c in cnt.items() if c >= 21}
            print(f"resume: {len(rows)} trials done, {len(done_parts)} parts complete -> skipped")
        except Exception:
            rows = []
    sel_cache = {}
    for rec in test_parts:
        if rec["part"] in done_parts:
            continue
        part = A.load_parsed_part(rec["path"])
        part["_scan_pts"], part["_scan_nrms"] = A.support_cloud(part)
        # exact normals: reuse trimmed labels to orient via PCA fallback
        from solver import estimate_normals as _en
        part["_scan_nrms"] = _en(part["_scan_pts"], radius=0.02 * part["D"], max_nn=30)
        tcad = A.TrimmedCAD(part)
        # A3 selection ONCE per part (dev-informed rule, frozen formula)
        if rec["part"] not in sel_cache:
            try:
                def val_fn(idx):
                    sub = A.TrimmedCAD_from_subset(tcad, idx)
                    rng = np.random.default_rng(7)
                    src, T_gt = synth_scan(sub, rng, 0.8, 1e-3)
                    if len(src) < 200:                 # degenerate viewpoint -> uncropped
                        src, T_gt = synth_scan(sub, rng, 1.0, 1e-3)
                    if len(src) < 200:
                        return 0.0, 1e-4               # give up scoring this subset
                    sn = _en(src, radius=0.02 * part["D"], max_nn=30)
                    T0 = np.eye(4)
                    T_s, _ = A.register_auto(sub, src, sn, 0.012 * part["D"], T0)
                    p = src @ T_s[:3, :3].T + T_s[:3, 3]
                    d, _ = sub.d_and_label(p)
                    return A.robust_score(d, 0.012 * part["D"]), 1e-4
                pick, table = A.select_auto(part, sigma=0.0024 * part["D"], val_score_fn=val_fn)
                sel_cache[rec["part"]] = pick
                print(f"{rec['part']}: A3 picked {pick['name']} (n={pick['n']}, U={pick['U']:.2g}, alarm={pick['alarm']})")
            except Exception as ex:
                sel_cache[rec["part"]] = dict(name="all", n=len(part["supports"]),
                                              U=float("nan"), alarm=True, idx=None,
                                              note=f"A3-selection fallback: {ex!r:.80}")
                print(f"{rec['part']}: A3 FALLBACK to all-supports ({ex!r:.80})")
        pick = sel_cache[rec["part"]]
        sel_idx = pick.get("idx") if pick["name"] != "all" else None
        run_part(rec, tcad, sel_idx, abs(hash(rec["part"])) % 10 ** 6, rows)
        # checkpoint after each part
        import json as _j
        _j.dump(rows, open(args.out + ".ckpt", "w"), default=float)
    import json as _j
    if os.path.exists(args.out):
        args.out += f".{int(time.time())}.json"
    _j.dump(dict(manifest_summary=dict(n_eligible=len(m["parts"]),
                                        n_ineligible=len(m["ineligible"])),
                 rows=rows), open(args.out, "w"), default=float, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
