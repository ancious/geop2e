"""Sanity test: analytical vs numerical Jacobian agreement for all primitives.
This is experiment E1 (validates UNIFIED_MATH.md + addresses reviewer R1#5 / R3
solver-level). If max|J_a - J_n|/||J_a|| is ~1e-6, the unified analytical form is
correct and matches the implementation convention (left perturbation, x_i scan frame).
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from features import PlaneFeature, CylinderFeature, SphereFeature

rng = np.random.default_rng(0)

# A non-trivial pose so x_i = R p_i + t is genuinely "scan frame".
T = np.eye(4)
T[:3, :3] = Rot.from_euler('yz', [-15, 5], degrees=True).as_matrix()
T[:3, 3] = [0.8, -1.5, 0.5]

feats = [
    PlaneFeature([0, 0, 1], 0.0),
    CylinderFeature([0, 0, 0], [0, 1, 0], 1.5),
    SphereFeature([4.0, 0.0, 2.0], 2.0),
]

print(f"{'primitive':10s} | {'max|Ja-Jn|':>12s} | {'rel err':>10s} | verdict")
print("-" * 60)
all_ok = True
for f in feats:
    pts = f.generate_points(2000, noise_std=0.0, extent=8.0, rng=rng)
    x = pts @ T[:3, :3].T + T[:3, 3]               # transform to scan frame
    Ja = f.compute_jacobian(x)                      # analytical (unified form)
    Jn = f.compute_numerical_jacobian(x)            # central difference, same convention
    abs_err = np.max(np.abs(Ja - Jn))
    scale = np.linalg.norm(Ja) + 1e-12
    rel = abs_err / scale
    ok = rel < 1e-4
    all_ok &= ok
    print(f"{f.kind:10s} | {abs_err:12.3e} | {rel:10.3e} | {'PASS' if ok else 'FAIL'}")

print("-" * 60)
print("ALL PASS" if all_ok else "SOME FAILED")
