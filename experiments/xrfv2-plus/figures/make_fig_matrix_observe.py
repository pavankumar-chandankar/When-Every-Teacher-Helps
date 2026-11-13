# Figures for the teacher x student matrix and the observability results.
# Matrix values: val_known_macro_f1 from runs_har/*/ JSON (pasted 2026-11-01), seeds 1-3.
# Observability: tracke/real_stage1_report.json (seed_results R_O, per-second scale);
# Track B error ratios: docs/TRACKB_H1_RESULT.md.
import re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e", "text.color": "#0b0b0b"})
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# ---------- teacher x student matrix ----------
txt = open("matrix_raw.txt").read()
M = "(?:imu|airpods|pose2d|pose3d|video|wifi)"
rows = re.findall(r"((?:imu|airpods|pose2d|pose3d|video|wifi|student)(?:_from_" + M + r")?)_seed([123]) \"val_known_macro_f1\": ([0-9.]+)", txt)
f1 = {(n, int(s)): float(v) for n, s, v in rows}
for s in (1, 2, 3):
    f1[("wifi_from_imu", s)] = f1[("student", s)]
mods = ["pose2d", "pose3d", "imu", "airpods", "video", "wifi"]
names = {"pose2d": "2D pose", "pose3d": "3D pose", "imu": "body IMU", "airpods": "AirPods", "video": "video", "wifi": "WiFi"}
base = {m: np.mean([f1[(m, s)] for s in (1, 2, 3)]) for m in mods}
G = np.full((6, 6), np.nan)
for i, st in enumerate(mods):
    for j, te in enumerate(mods):
        if st != te:
            G[i, j] = 100 * (np.mean([f1[(f"{st}_from_{te}", s)] for s in (1, 2, 3)]) - base[st])
cmap = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
cmap.set_bad("#f1f0ed")
fig, ax = plt.subplots(figsize=(3.5, 2.75))
im = ax.imshow(G, cmap=cmap, vmin=0, vmax=9.5)
for i in range(6):
    for j in range(6):
        if i == j:
            ax.text(j, i, "–", ha="center", va="center", color=INK2, fontsize=7)
        else:
            v = G[i, j]
            ax.text(j, i, f"+{v:.1f}", ha="center", va="center", fontsize=6.8, color="white" if v > 5.2 else INK)
short = {"pose2d": "2D\npose", "pose3d": "3D\npose", "imu": "body\nIMU", "airpods": "Air-\nPods", "video": "\nvideo", "wifi": "\nWiFi"}
ax.set_xticks(range(6), [f"{short[m]}\n{base[m]:.2f}" for m in mods])
ax.set_yticks(range(6), [f"{names[m]} ({base[m]:.2f})" for m in mods])
ax.set_xlabel("Teacher (its own macro-F1)")
ax.set_ylabel("Student (macro-F1 alone)")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks(np.arange(-.5, 6, 1), minor=True); ax.set_yticks(np.arange(-.5, 6, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Gain (macro-F1 points)", fontsize=7); cb.ax.tick_params(labelsize=6.5, length=2); cb.outline.set_visible(False)
fig.tight_layout()
fig.savefig("fig_matrix.pdf"); fig.savefig("fig_matrix.png", dpi=300); plt.close(fig)

# ---------- observability ----------
RO = {  # per-second R_O, seeds 1-3 (wrist = F1, body = F3)
    "2D pose": ([0.44458, 0.47867, 0.46396], [0.39934, 0.41869, 0.41859]),
    "3D pose": ([0.51065, 0.57953, 0.53802], [0.44570, 0.52958, 0.45717]),
    "video":   ([0.84655, 0.79449, 0.80938], [0.59976, 0.57903, 0.58169]),
    "WiFi":    ([0.90823, 0.92801, 0.92432], [0.74649, 0.81302, 0.77775]),
}
CTRL = [0.51061, 0.51708, 0.50500]  # AirPods IMU -> head motion (glasses), positive control
R_B = {"gyro\nnorm": [1.150, 1.395, 1.081], "accel.\nenergy": [1.398, 1.498, 1.282], "energy\nchange": [1.005, 1.008, 1.005]}
WR, BO = "#6250d6", "#1baf7a"
fig, (a, b) = plt.subplots(1, 2, figsize=(3.5, 1.95), gridspec_kw={"width_ratios": [1.45, 1]})
xs = np.arange(len(RO))
for k, (sensor, (w, bd)) in enumerate(RO.items()):
    for v in w:
        a.scatter(k - 0.13, v, s=14, color=WR, edgecolor="white", linewidth=0.6, zorder=3)
    for v in bd:
        a.scatter(k + 0.13, v, s=14, color=BO, marker="D", edgecolor="white", linewidth=0.6, zorder=3)
for v in CTRL:
    a.scatter(4, v, s=14, color="#8a8984", marker="s", edgecolor="white", linewidth=0.6, zorder=3)
a.axvline(3.55, color=GRID, lw=0.8)
a.axhline(1.0, color=INK2, lw=0.8)
a.text(-0.45, 1.015, "activity label only", fontsize=6.3, color=INK2, va="bottom")
a.set_xticks(list(xs) + [4], ["2D\npose", "3D\npose", "video", "WiFi", "AirPods\n(control)"]); a.set_ylim(0.2, 1.1); a.set_xlim(-0.5, 4.45)
a.set_ylabel("Error vs. label-only ($R_O$)")
a.scatter([], [], s=14, color=WR, label="wrist motion"); a.scatter([], [], s=14, color=BO, marker="D", label="body motion")
a.legend(loc="lower right", ncol=2, columnspacing=0.8, fontsize=6.3, frameon=False, handletextpad=0.2, borderaxespad=0.1)
a.set_title("(a) Can the sensor see it?", loc="left", fontsize=7.3)
for k, (lab, vals) in enumerate(R_B.items()):
    for o, v in zip([-0.12, 0, 0.12], vals):
        b.scatter(k + o, v, s=14, color="#2a78d6", edgecolor="white", linewidth=0.6, zorder=3)
b.axhline(1.0, color=INK2, lw=0.8); b.axhline(0.90, color=INK, lw=1.0)
b.text(2.45, 0.91, "pass (\u2264 0.90)", fontsize=6.3, ha="right", va="bottom")
b.set_xticks(range(3), list(R_B.keys())); b.set_ylim(0.84, 1.56); b.set_xlim(-0.5, 2.5)
b.set_ylabel("Error ratio $R$")
b.set_title("(b) WiFi \u2192 IMU motion", loc="left", fontsize=7.3)
for ax_ in (a, b):
    for s in ("top", "right"):
        ax_.spines[s].set_visible(False)
    ax_.yaxis.grid(True, color=GRID, lw=0.5); ax_.set_axisbelow(True); ax_.tick_params(length=2, width=0.6)
fig.tight_layout(w_pad=0.6)
fig.savefig("fig_observe.pdf"); fig.savefig("fig_observe.png", dpi=300); plt.close(fig)
print("ok")
