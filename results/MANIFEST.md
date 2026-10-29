# code/minimal/results — canonical artifacts manifest (2026-08-27)

Per final-verification prescriptions, the SUFFIXED file is canonical where noted.

| Experiment | Canonical file | Notes |
|---|---|---|
| E3 full ablation | e3_hv_ablation_full.json | 320 paired registrations; 3 gpt-review rounds passed pre-run |
| E3b forced-trigger | e3b_forced_trigger.json (timestamped variant if present: newest) | 60 forced runs vs recorded bypass |
| E3c diagnostic | e3c_loss0_diagnostic.json (newest timestamped) | corrected frozen-association analysis; e3c_run.log.superseded marks the WRONG +1deg interpretation run |
| E3d reanalysis | e3d_reanalysis.json (newest timestamped) | raw-vs-sym endpoints, activation counts, per-cell runtime, E5 |
| E2 canonical table | **e2_controlled_table.json.794e0f23.json** (SUFFIXED = canonical; unsuffixed e2_controlled_table.json is STALE) | canonical FGR sha per scan: S1 80910149ec664bfa, S2 c350cb3fed7b8dcf, S3 f8b7f774aa27ee4d; field `canonical_fgr_persisted` read as loaded-from-cache semantics (false = newly created & persisted this run, NOT unpersisted) |
| E4 Cranfield | e4_cranfield.json produced by e4_run5.log (frame-corrected rerun) | runs 2-4 superseded (frame defect: raw-STL primitives vs prepared template — excluded from interpretation); run5 = single persisted coordinate frame |
| RPMNet adapters | e1c_rpmnet_smoke.json.* / e2_rpmnet.json.* / e4_rpmnet.* | zero-shot partial_0.7 checkpoint |
