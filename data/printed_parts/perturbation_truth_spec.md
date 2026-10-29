# E8e 扰动真值再生规格（50×5 试验，CAD 系）
生成代码 = geop2e/code/minimal/e8e_bench.py main() 试验循环，确定性序列：
- rng = np.random.default_rng(SEED0 + si*1000)，SEED0=20260905，si∈{1..5}（PARTS 序）
- 每个 trial（每零件前 10 个 = intact，e8e）按序消费 rng：
  1) ax = rng.normal(size=3); ax /= norm(ax)
  2) ang = rng.uniform(15.0, 60.0)          # 度
  3) tg = rng.uniform(-1, 1, 3) * 0.08 * ext # ext=零件支撑云对角线（各件 fim_diagnostic.json 有 T_ref/无 ext，用 CAD 基元包围盒对角线：见 primitives JSON 各面 guide 点范围；或直接用 e8e 源码同式）
- G = [Rot.from_rotvec(ax*deg2rad(ang)) | tg]；扰动点云 pts' = pts @ Rg.T + tg
- 期望位姿 T_exp = T_ref @ inv(G)；方法位姿 T̂ 的误差 E = T̂ @ inv(T_exp)（CAD 系）
- e8h（缺损）每件 20 试验（前 10=保留30%、后10=保留50%），rng 序列在上述三步后追加：ca=rng.normal(3)/norm、q=quantile(pts_p@ca, ret)、ret 按 tr//10 取 0.7/0.5
数据源 JSON：e8e_bench.1788611034.json（每试验含 T 矩阵=rre/tte 所用 T̂）+ e8h_defect.json
注意：两文件 rng 消费序不同（e8h 每 trial 额外消费 ca），不可混用序列。
