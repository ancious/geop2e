"""E1c — zero-shot RPMNet adapter (partial-to-complete representative, R1's demand).

Runs in the `guji` env (torch 2.9; NO open3d there) from the Cranfield repo root
so `src.toolboxes.learning3d` resolves. Reads .npz pairs {src: Nx3 mm, tgt: Nx3 mm}
produced by the geo-env drivers, runs the bundled partial_0.7-noisy RPMNet
checkpoint with fixed preprocessing, writes {pair_id: {T (4x4 world frame),
time_s, n_src, n_tgt}} JSON. Zero training, no test tuning; pretrained-domain
mismatch is disclosed in every consuming table.

Preprocessing (disclosed): per-pair centering to the SOURCE centroid offset by
target-centroid alignment is NOT applied (that would solve translation); instead
each cloud is centered on its own centroid and both are scaled by
1/extent(target) (a rigid similarity applied identically to the pair, inverted
in the returned transform). Fixed 1024 points per cloud by seeded uniform
subsampling; normals by local PCA (radius = 2% extent, orient toward centroid),
matching Learning3D's expected Nx6 input.

Usage (from cranfield repo root):
  ~/miniconda3/envs/guji/bin/python <this file> --pairs <dir> --out <json>
"""
import os, sys, json, time, argparse
import numpy as np
import torch
from scipy.spatial import cKDTree

CRANFIELD = "data/cranfield"
L3D = os.path.expanduser("~/tmp/builds/learning3d")   # cloned vinits5/learning3d (checkpoints live in CRANFIELD)
sys.path.insert(0, CRANFIELD)
sys.path.insert(0, os.path.dirname(L3D))              # import as PACKAGE (relative imports)

# torch_scatter shim: learning3d/__init__ imports ALL models incl. IAE which needs
# the compiled torch_scatter extension (only scatter_mean/scatter_max at import time;
# RPMNet/PPFNet do not call it). Pure-torch implementations via scatter_reduce.
import types
import torch as _torch
_ts = types.ModuleType("torch_scatter")
def _dim_size(index, dim, dim_size):
    if dim_size is not None:
        return dim_size
    return int(index.max().item()) + 1 if index.numel() else 0
def scatter_mean(src, index, dim=0, dim_size=None, **kw):
    n = _dim_size(index, dim, dim_size)
    out = _torch.zeros(n, *src.shape[1:], dtype=src.dtype, device=src.device)
    idx = index.reshape(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_reduce_(dim, idx, src, reduce="mean", include_self=False)
    return out
def scatter_max(src, index, dim=0, dim_size=None, **kw):
    n = _dim_size(index, dim, dim_size)
    out = _torch.full((n, *src.shape[1:]), float("-inf"), dtype=src.dtype, device=src.device)
    idx = index.reshape(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out.scatter_reduce_(dim, idx, src, reduce="amax", include_self=False)
    return out, None
_ts.scatter_mean = scatter_mean
_ts.scatter_max = scatter_max
sys.modules.setdefault("torch_scatter", _ts)

from learning3d.models import RPMNet, PPFNet  # noqa: E402

CKPT = os.path.join(CRANFIELD, "src/toolboxes/learning3d/checkpoints/"
                    "exp_rpmnet_partial_0.7_noisy_0.01_floor/models/best_model_snap.t7")
N_PTS = 1024
SEED = 20260827


def pca_normals(pts, radius, max_nn=30):
    tree = cKDTree(pts)
    centroid = pts.mean(0)
    normals = np.zeros_like(pts)
    for i, p in enumerate(pts):
        idx = tree.query_ball_point(p, r=radius)
        if len(idx) < 4:
            idx = tree.query(p, k=min(max_nn, len(pts)))[1]
            idx = np.atleast_1d(idx)
        nn = pts[idx] - p
        _, vecs = np.linalg.eigh(nn.T @ nn)
        n = vecs[:, 0]
        if np.dot(n, centroid - p) < 0:
            n = -n
        normals[i] = n
    return normals


def to_fixed_n(pts, rng):
    if len(pts) >= N_PTS:
        idx = rng.choice(len(pts), N_PTS, replace=False)
    else:
        idx = rng.choice(len(pts), N_PTS, replace=True)
    return pts[idx]


def run_pair(model, src, tgt, device):
    rng = np.random.default_rng(SEED)
    tp0 = time.time()
    ext = float(np.linalg.norm(np.ptp(tgt, axis=0)))
    c_s, c_t = src.mean(0), tgt.mean(0)
    s = 1.0 / ext
    src_n = to_fixed_n((src - c_s) * s, rng)
    tgt_n = to_fixed_n((tgt - c_t) * s, rng)
    sn = pca_normals(src_n, radius=0.05, max_nn=30)
    tn = pca_normals(tgt_n, radius=0.05, max_nn=30)
    S = np.concatenate([src_n, sn], 1).astype(np.float32)
    T = np.concatenate([tgt_n, tn], 1).astype(np.float32)
    t_preproc = time.time() - tp0
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        model.eval()
        out = model(torch.from_numpy(T)[None].to(device),
                    torch.from_numpy(S)[None].to(device))
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    # learning3d RPMNet forward returns a dict; the cranfield tester uses
    # output['est_T'] -> (1,4,4) estimated transformation template<-source
    Tn = out["est_T"].detach().cpu().numpy()[0].astype(np.float64)
    # denormalise: world transform maps original src -> original tgt
    # normalized frame: x' = s(x - c);  T_world = A^-1 @ Tn @ A,  A = [sI, -s*c_t]... src frame
    A = np.eye(4); A[:3, :3] = s * np.eye(3); A[:3, 3] = -s * c_s
    B = np.eye(4); B[:3, :3] = s * np.eye(3); B[:3, 3] = -s * c_t
    Ainv = np.eye(4); Ainv[:3, :3] = np.eye(3) / s; Ainv[:3, 3] = c_t  # maps norm -> tgt world
    # T_world(src->tgt) = denorm_tgt(Tn(nomralized src->normalized tgt))
    # x_tgt_world = (Tn @ (s(x - c_s)) ) / s + c_t
    W = np.eye(4)
    W[:3, :3] = Tn[:3, :3]
    W[:3, 3] = (Tn[:3, 3] / s) + c_t - (Tn[:3, :3] @ c_s)
    return W, float(dt), dict(extent=ext, scale=s, n_src=len(src), n_tgt=len(tgt),
                              preproc_s=float(t_preproc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, help="dir with *.npz {src,tgt}")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RPMNet(feature_model=PPFNet()).to(device)
    ck = torch.load(CKPT, map_location="cpu")
    sd = ck["model"] if "model" in ck else ck
    # the cranfield-era checkpoint names the feature extractor 'feature_model';
    # current learning3d names it 'feat_extractor' -> remap
    sd = { (k.replace("feature_model.", "feat_extractor.") if k.startswith("feature_model.") else k): v
           for k, v in sd.items() }
    model.load_state_dict(sd)
    print(f"loaded {CKPT} on {device}")
    results = {}
    for f in sorted(os.listdir(args.pairs)):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(args.pairs, f))
        try:
            W, dt, rec = run_pair(model, z["src"], z["tgt"], device)
            results[f[:-4]] = {"T": W.tolist(), "time_s": dt, **rec}
            print(f"{f[:-4]}: ok ({dt:.2f}s)")
        except Exception as e:
            results[f[:-4]] = {"error": repr(e)[:300]}
            print(f"{f[:-4]}: ERROR {e!r}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        args.out += f".{int(time.time())}.json"
    json.dump({"ckpt": CKPT, "n_pts": N_PTS, "seed": SEED, "device": str(device),
               "note": "zero-shot, partial_0.7_noisy checkpoint, own-centroid + "
                       "target-extent normalisation, fixed-1024 seeded subsample",
               "results": results}, open(args.out, "w"), indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
