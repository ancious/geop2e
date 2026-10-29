# E8e 门审记录（2026-09-05）
- 主审：GPT-5.6-sol xhigh（重试成功；此前两次挂起 + Opus5 网关空响应 ×2 —— 通道故障窗口，非配额）。
- 判定：FIX-FIRST(crash_risk, target_fairness, calib_fairness, reproducibility)。
- 修复（全部落地）：
  F1 registration_gicp 不存在于 open3d 0.19 → registration_icp + TransformationEstimationForGeneralizedICP（CRITICAL）
  F2 PCA 候选 12→24（补全 8 符号模式 × det 过滤）
  F3 trimmed_icp 目标改为与 open3d 方法相同的 1mm STEP 云（公平性）
  F4 geop2e_cal → geop2e_cal_oracle（同零件调参泄漏显式标注）
  F5 JSON 断点顺序（零件插入后再 dump，末零件不丢）
  F6 评估云 1M 采样缓存 npz（跨运行确定性）
  F7 voxel 后法向归一化
  F8 protocol 块补齐（预处理参数/TTE 语义/种子）
- PASS 项：pose_math（T_exp=T_ref@inv(G) 代数验证）、normal_rotation、voxel-normal 存活。
- 注：Opus 交叉核验未完成（网关故障）→ 本次门为 GPT 单审 + 确定性自验；通道恢复后可对结果 JSON 复审。
