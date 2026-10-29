# §4.5 定稿文本（D4，图 /tmp/fig8.png /tmp/fig9.png 已生成）

**勘误 2026-09-08**：本文原写 AUC=0.983，系原始绘图脚本 ROC 标签构造的 tuple-key bug（`for p, v in certs.items()` 遍历元组键 → 全部 unit 被标 fail_rate=1.0 → 退化 ROC，`np.trapezoid` 积出 (29/30+1)/2=0.98333）。正确值：**0.92**（含边界口径，正类 = cell success rate ≤ 50%，11/30 阳性；严格 <50% 口径为 0.89，两边界 unit 摆动）。取证与复现：~/tmp/figrecon/；记忆：geop2e-fig12-auc-defect。

**4.5 Operating Envelope under Single-View Occlusion: Observability Diagnosis and Profile-Guided Fallback**

[P1] Section 4.4 validated GeoP2E under full-coverage scanning. Industrial single-camera or fixtured acquisition, however, yields partial visibility; this subsection deliberately evaluates beyond the target acquisition regime (single viewpoint, point retention 0.20-0.55, initial misalignment 10-30 deg, translations 3-10 mm) in order to delineate the operating envelope rather than to claim robustness superiority.

[P2] Protocol. Five parts x six viewpoints x ten perturbations x fifteen method arms (4,500 registrations). All primitives are parsed directly from the STEP B-Rep (no point-cloud reconstruction); viewpoint visibility uses exact mesh ray casting; evaluation uses a symmetry-aware whole-model pose error on an independent evaluation cloud, with certified symmetry groups (including a strict C6 for the coaxial bundle). The primary hypothesis H1 and seven directional predictions were frozen before the run; hits and misses are both reported.

[P3] H1 was rejected: pooled GeoP2E-all attained 47.3% versus 46.0% for matched-robustness ICP (+1.3 pp; cluster-bootstrap 95% CI [-18.7%, +21.3%]), and the strongest arm under this regime is budget-matched multi-start point-to-triangle ICP (78.0% at 10.2 s). We do not claim robustness superiority under single-view occlusion.

[P4] The failure, however, is one-dimensional. Splitting by dominant primitive class, on the three plane-dominant parts (180 runs) the bounded GeoP2E formulation attains 65% at 1.4 s — statistically indistinguishable from dense point-to-triangle ICP (63.3%) and above gated ICP (45.0%); on the two cylinder-dominant parts the unbounded formulation collapses (21.7%) while the bounded formulation recovers to 52.5%.

[P5] In consequence, with roughly ten analytic CAD primitives in place of ~10^5 mesh triangles, the formulation retains dense-mesh-level success on plane-dominant geometry under severe occlusion (117/180 vs 114/180), and its successful-trial median whole-model error (0.13-0.19 mm) is consistent with the 0.25-0.28 mm reported in Sec. 4.2 — the accuracy statement independently reproduces on new parts under a harder condition.

[P6] Mechanism I: the observability certificate predicts failure a priori. The smallest eigenvalue of the Fisher information matrix built from the visible primitive subset separates failing from succeeding (part, viewpoint) cells with in-sample AUC = 0.92 (positive class: cell success rate at or below 50%; Fig. 8); the stepped-shaft part is declared unsolvable in advance (axial rotation unconstrained by any visible feature) — a capability the ICP family has no counterpart of.

[P7] Mechanism II: infinite-primitive cross-face attraction, and its repair. Failure-sample error histograms show discrete spikes at 9.1 mm and 14.1 mm — exactly the CAD inter-feature pitches (9.1 mm shoulder offset; 14.095 mm end-plane gap, smaller than the 15 mm coarse search radius). The bounded-face correspondence test (Sec. 3, a Phase-1 profile switch) removes these basins: on full scans the switch leaves results unchanged or improved (49/50; medians 0.006-0.141 mm, versus 0/10 collapse on the thin plate without it), and under occlusion lifts GeoP2E from 47.3% to 60.3% pooled and from 23/60 to 52/60 on the coaxial bundle — the best result of any arm on that part.

[P8] Mechanism III: observability selection is pathological under occlusion. Frozen E-opt-8 (22%), occlusion-recomputed selection (23%), and FSV-weighted multi-start selection (44%, including a catastrophic 0/60 on the coaxial bundle) are all strictly inferior to the full primitive set — jointly with the thin-plate trap of Table 5 this supports one rule: observability is a good online diagnostic and a bad offline selector under partial visibility; subset selection belongs in Phase 1 with an all-primitive fallback.

[P9] Deployment consequence: profile-guided fallback. Gating each (part, viewpoint) cell by the offline certificate (threshold calibrated leave-one-part-out) and falling back to the multi-start baseline only on flagged cells attains 70.0% pooled success at a mean runtime of 2.2 s (multi-start everywhere: 78.0% at 10.2 s), recovering most of the gap at one quarter of the cost, with every fallback tagged by a named unobservable degree of freedom.

[P10] Boundary and open problems. The current envelope is single-view occlusion with 10-30 deg initial error; cylinder-dominant geometry at the most severe retention remains the boundary (12/60 on the stepped shaft even with the bounded switch — matching the certificate's unsolvable verdict). Trimmed ICP, re-tuned over thresholds {10, 20, 40} mm, remains at 1-2% under this regime. A hard one-point-one-face assignment is left to future work.

[P11] Negative results reported as such: pre-registered predictions hit 2/7; the visible-side drift hypothesis (omega > 3 for baselines) was falsified (omega ~ 1: baseline failure is an initialization-basin phenomenon), changing the engineering advice from "invest in weights" to "invest in initialization."

## 附: 其他改动点
- §3 半页: bounded-face correspondence test 形式化(切向投影距离 + 阶段调度容差 max(2.0, 0.25*radius)) + λ_min certificate 定义
- §5 一段: operating envelope 表(ρ/几何类/初值) + 部署决策规则
- §1 贡献 +1 条: observability-certified operating envelope + profile-guided fallback (4,500-次预注册压力测试)
- Table 6 = D3 四臂×5件+几何类池化表; Table 7 = 每件 McNemar + λ_min + flagged; Fig 8/9 已生成
- D6 回复信 R1 条目: "Single-view occlusion stress test on five non-gear parts: 4,500 pre-registered registrations across 15 method arms, primary hypothesis rejected and reported as such; a bounded-face profile switch repairing the identified cross-face failure mechanism (coaxial bundle 23->52 of 60, best of all arms); and an observability certificate (in-sample AUC 0.92) that flags unsolvable configurations a priori."
