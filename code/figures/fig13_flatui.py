import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman", "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(5.4, 3.9))
arms = {"B2": (64.0, 1.1), "B3": (46.0, 2.2), "B6": (67.3, 0.8), "B10b": (78.0, 10.2),
        "GeoP2E\n(pre-reg.)": (47.3, 1.0), "GeoP2E\n(bounded)": (60.3, 1.4), "hybrid": (70.0, 2.2)}
for k, (s_, t_) in arms.items():
    c = "#2980b9" if k in ("GeoP2E\n(bounded)", "hybrid") else ("#27ae60" if k in ("B2","B3","B6","B10b") else "#8e44ad")  # flat-UI: GeoP2E=blue, baselines=green, pre-reg ablation=purple
    ax.scatter(t_, s_, s=150 if k=="hybrid" else 80, marker=("*" if k=="hybrid" else ("^" if "Geo" in k else "s")), c=c, zorder=3)
    if k == "B10b":   # only the top-right one moves to bottom-left (was overflowing)
        ax.annotate(k, (t_, s_), textcoords="offset points", xytext=(-8, -12), fontsize=9, ha="right")
    else:             # all others stay at top-right
        ax.annotate(k, (t_, s_), textcoords="offset points", xytext=(6, 5), fontsize=9, ha="left")
ax.set_xlim(0, 12); ax.set_ylim(38, 88)
ax.set_xlabel("mean runtime per registration (s)")
ax.set_ylabel("success rate (%)")
ax.set_title("Success vs runtime under single-view occlusion", fontsize=11)
ax.grid(alpha=.25)
fig.savefig("fig9_regen300.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
print("fig13 regenerated: only B10b bottom-left, others top-right")
