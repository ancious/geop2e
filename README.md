# GeoP2E — Reproducibility Package

Code, configurations, random seeds, and the numerical data underlying the principal tables and figures of:

> **GeoP2E: Explicit Geometric Feature Matching via Point-to-Equation Registration**
> (Results in Engineering, manuscript RINENG-D-26-20091, revision 1)

GeoP2E registers scanned point clouds of incomplete/defective industrial parts to analytical
CAD primitives (planes, cylinders, spheres) using Fisher-information-based observability
selection, analytical Jacobians on SE(3), and geometric hypothesis verification.

## Layout

| Path | Contents |
|---|---|
| `code/` | Core library (feature/selection/solver/baselines/hypothesis) |
| `code/minimal/` | Experiment drivers E0–E9 (see table map below) |
| `code/figures/` | Figure generators (Fig. 12, Fig. 13) |
| `results/` | Archived experiment outputs (JSON/JSONL/Markdown) — the numbers behind the paper |
| `results/docs/` | Experiment-conclusion notes (E9 conclusions, D45 selection study) |
| `data/printed_parts/` | Processed scans (100k-point `.npz`), parsed CAD primitives (`.json`), FIM diagnostics, symmetry groups, and a manifest for the five SLA-printed validation parts |
| `data/cad_printed_parts/` | Fine STL meshes + scaled STEP models of the printed parts |

Large raw artifacts (original vendor scans of the printed parts, E8/E9 evaluation meshes,
RPM-Net pair archives; ~2.4 GB) are **not** in this repository — see
`large_files_manifest.md` for exact files, sizes, and where they will be archived.

Real industrial **gear scans** contain project-specific inspection information and are
available from the corresponding author upon reasonable request, subject to institutional
approval (see the manuscript's Data availability statement).

## Environment

Python 3.13, NumPy 2.4.3, SciPy, Open3D (ICP/GICP/Trimmed ICP baselines), and — for the
RPM-Net zero-shot baseline only — `learning3d` with PyTorch. CAD parsing uses an OCCT/OCP
build. Experiments were run on CPU (Ubuntu 24.04, i7-12700K, 21 GB RAM).

## Data map (paper item → artifacts)

| Paper item | Source artifacts | Reproduce with |
|---|---|---|
| Table 1 (ABC benchmark) | `results/` E-benchmark JSONs (see `results/MANIFEST.md`) | `code/experiments.py` |
| Table 2 / tuned gear pipeline | `results/EXPERIMENT_RESULTS_MINIMAL.md` §gear | `code/real_gear_experiment.py` |
| Table 3 + Table A5 (common-protocol baselines) | `results/e2_controlled_table*.json` (canonical: suffixed `…json.794e0f23.json` per `results/MANIFEST.md`) | `code/minimal/e2_controlled_table.py` |
| Table 4 / 5 (printed parts, crop stress) | `results/e8e_bench*.json`, `results/e8h_defect*.json`, `results/E8_FIRSTPASS_REPORT.md`; part data in `data/printed_parts/` | `code/minimal/e8e_bench.py`, `e8h_defect.py` |
| Table 6 / 7, Fig. 12 (occlusion stress test, 4,500 runs) | `results/e9/analysis.1.json` (**main 4,500-run analysis** — `analysis.json` is a null-test), `results/e9/focus.4.jsonl` (deployed GB arm, 300 trials), `results/e9/d1_gate.json` (λ_min certificates + LOPO), `results/e9/main.jsonl` (G1 failures for panel (c)) | `code/minimal/e9_viewpoint_occlusion.py`; figure: `python code/figures/fig12_gpt6.py` (reads `results/e9/fig12_data.json`, regenerated from `focus.4.jsonl` + `d1_gate.json`; in-sample AUC = 0.9234 → "0.92") |
| Fig. 13 (success–runtime Pareto) | `results/e9/analysis.1.json` aggregates | `code/figures/fig13_flatui.py` (aggregates currently hardcoded — TODO: parametrize) |
| HV ablation (Sec. 4.5 / response letter) | `results/e3_hv_ablation_full.json`, `results/e3b_forced_trigger.json` | `code/minimal/e3_hv_ablation.py`, `e3b_forced_trigger.py` |
| Cranfield external benchmark | `results/e4_cranfield.json` | `code/minimal/e4_cranfield.py` (dataset path: set `CRANFIELD` to your local copy of the public Cranfield benchmark) |

Random seeds are fixed inside each driver (`rng_seed`, per-trial stimulus seeds derived by
stable hashing; see `trial_seed` in `code/minimal/e3_hv_ablation.py` and the E9 protocol in
`code/minimal/e9_viewpoint_occlusion.py`).

## Notes

- `data/gear_step_cad/` referenced by gear drivers is the authors' local dump of gear STEP
  CAD + scan files (industrial data, on request); the drivers degrade gracefully if absent.
- The E9 protocol had two recorded amendments (two-stage radius 15→3 mm; mesh-raycast
  primary visibility), disclosed in the manuscript's Supplementary material.
- License: MIT (see `LICENSE`). Citation entry: DOI to be added upon publication.
