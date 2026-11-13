# Single-panel room figure (unseen-room accuracy), for use once the teaser shows the cosine result.
# Numbers from domain_vs_activity_summary.json (leave_one_env_out_knn).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "pdf.fonttype": 42, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b", "hatch.linewidth": 0.6})
WIFI, IMU, INK, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#e4e3df"
acc_wifi = np.array([0.277989337395278, 0.17884130982367757, 0.2236084452975048]) * 100
acc_imu = np.array([0.9131759329779132, 0.8627204030226701, 0.8857965451055663]) * 100
acc_maj = np.array([0.19497334348819498, 0.16120906801007556, 0.16698656429942418]) * 100
fig, b = plt.subplots(figsize=(3.5, 1.55))
w, gap, x = 0.32, 0.03, np.arange(3)
b.bar(x - (w + gap) / 2, acc_wifi, w, color=WIFI, linewidth=0, label="WiFi")
b.bar(x + (w + gap) / 2, acc_imu, w, color=IMU, hatch="////", edgecolor="white", linewidth=0, label="IMU")
for i in range(3):
    b.plot([i - w - gap, i + w + gap], [acc_maj[i]] * 2, color=INK, lw=1.2, solid_capstyle="butt",
           label="most frequent class" if i == 0 else None)
    b.text(i - (w + gap) / 2, acc_wifi[i] + 2.5, f"{acc_wifi[i]:.0f}", ha="center", va="bottom", fontsize=6.5)
    b.text(i + (w + gap) / 2, acc_imu[i] + 2.5, f"{acc_imu[i]:.0f}", ha="center", va="bottom", fontsize=6.5)
b.set_xticks(x, ["dining room", "study room", "bedroom"])
b.set_ylim(0, 108); b.set_yticks([0, 25, 50, 75, 100])
b.set_ylabel("Accuracy in\nunseen room (%)")
for s in ("top", "right"):
    b.spines[s].set_visible(False)
b.yaxis.grid(True, color=GRID, lw=0.5); b.set_axisbelow(True); b.tick_params(length=2, width=0.6)
h, l = b.get_legend_handles_labels()
order = [l.index("WiFi"), l.index("IMU"), l.index("most frequent class")]
b.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, frameon=False, handlelength=1.2)
fig.tight_layout()
fig.savefig("fig_room_unseen.pdf"); fig.savefig("fig_room_unseen.png", dpi=300)
print("ok")
