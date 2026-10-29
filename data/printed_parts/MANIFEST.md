# reglaw 共享包 — 薄板 21644（2026-09-06）

相对本目录（geop2e/data/reglaw_share/）：

| 文件 | 内容 |
|---|---|
| `plate21644_fim_diagnostic.json` | 离线 CAD-FIM 与 T_ref 处 active-FIM 全谱（特征值+特征向量+板法向投影）、对应点按类型计数、**λ 语义说明**（未加权点到方程 Jacobian、rot 无量纲×trans mm 混合、未做特征长度归一） |
| `plate21644_calibration.json` | 该件基准标定档（all-primitives, lam=0.1, ang=30°, r=3.0mm）+ 溯源（同零件调参=oracle 已披露） |
| `plate21644_scan_100k.npz` | 扫描 10 万点子样本（points+normals，扫描器坐标系，mm，seed=20260906，源自清理后 0.5mm 体素工作云） |
| `plate21644_cad_primitives.json` | 13 个解析基元参数（平面 n·x+d=0 / 圆柱 axis_point+axis+r / 球 c+r，CAD 系 mm）+ T_ref（scan→CAD 密集 ICP 锚点）+ 参考 p2m |

P3（逐试验横评 JSON，本机直接读）：
- `geop2e/code/minimal/results/e8e_bench.1788611034.json`（完整场景 8 方法 × 50 试验，含每试验 T 矩阵/p2m/RRE）
- `geop2e/code/minimal/results/e8h_defect.json`（缺损场景；注意 retention 标签语义反转：label 0.7=实际保留 30%、0.5=保留 50%）
- `geop2e/code/minimal/results/e8e_bench.1788611034_scored.json`（e8g 评分汇总）

λ 对齐注意：我们的 H 无加权、混合单位（rot/trans）——你们的闭式律若用特征长度归一化，需先归一或按各自语义对照；板法向（CAD 系）= 最小 PCA 轴，在诊断 JSON 里有。
