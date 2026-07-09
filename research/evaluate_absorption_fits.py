"""
Evaluate and overlay c = f(H_src_norm) fits across absorption levels.

For each experiment folder in `mc_files`, loads:
  - results.json                     -> scatter points (h_src_norm, optimal_c, min_rmse)
  - c_prediction_model_params.json   -> fitted linear/poly/power coefficients

Figure (same aesthetic as predict_c_from_proximity.py, right panel):
  - scatter + bold red fit line come from MAIN_CONFIG
  - every absorption's fit line is overlaid (legend = alpha + equation + R^2)

Run predict_c_from_proximity.py on each folder first so the params JSON exists;
if it is missing, the fit is computed on the fly from results.json.
"""

import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
EXP_DIR = os.path.join(project_root, "results", "monte_carlo_experiments")

# ===================================================================== CONFIG
mc_files = ["mcfix_a0.1","mcfix_a0.2", "mcfix_a0.3", "mcfix_a0.4", "mcfix_a0.5", "mcfix_a0.6"]   # folders to overlay (add 0.1/0.2/0.5/0.6 when ready)
MAIN_CONFIG = "mcfix_a0.4"                  # scatter + bold line from this folder
MODEL = "linear"                            # "linear" | "polynomial" | "power"
SAVE = True
OUT = os.path.join(EXP_DIR, "absorption_fits_overlay.png")
# ===========================================================================


def absorption_label(results, folder):
    a = results[0].get("absorption")
    if a is not None:
        return float(a)
    m = re.search(r"a(\d+\.?\d*)", folder)   # e.g. "mcfix_a0.3" -> 0.3
    return float(m.group(1)) if m else folder


def fit_line(params, model, H, C):
    """Return (predict_fn, r2, equation) from saved params, or fit on the fly."""
    if params is not None and model in params.get("models", {}):
        m = params["models"][model]
        if model == "linear":
            fn = lambda h: m["intercept"] + m["coefficient"] * h
        elif model == "polynomial":
            fn = lambda h: m["intercept"] + m["coef_h"] * h + m["coef_h2"] * h ** 2
        else:
            fn = lambda h: m["a"] * h ** m["b"]
        return fn, m["r2_score"], m["equation"]
    # fallback: on-the-fly linear fit
    s, i = np.polyfit(H, C, 1)
    r2 = 1 - np.sum((C - (s * H + i)) ** 2) / np.sum((C - C.mean()) ** 2)
    return (lambda h: i + s * h), r2, f"C = {i:.2f} + {s:.2f} * H_src_norm"


def load(folder):
    base = os.path.join(EXP_DIR, folder)
    results = json.load(open(os.path.join(base, "results.json")))
    pj = os.path.join(base, "c_prediction_model_params.json")
    params = json.load(open(pj)) if os.path.exists(pj) else None
    H = np.array([r["h_src_norm"] for r in results])
    C = np.array([r["optimal_c"] for r in results])
    RMSE = np.array([r["min_rmse"] for r in results])
    runs = {r["run_id"]: r["optimal_c"] for r in results}
    fn, r2, eq = fit_line(params, MODEL, H, C)
    return {"alpha": absorption_label(results, folder), "H": H, "C": C, "RMSE": RMSE,
            "runs": runs, "fn": fn, "r2": r2, "eq": eq, "n": len(results)}

loaded = {f: load(f) for f in mc_files}

# ----------------------------------------------------------------- EVALUATION
print(f"{'folder':<14}{'alpha':>6}{'n':>5}{'R2':>8}   equation")
for f in mc_files:
    d = loaded[f]
    print(f"{f:<14}{str(d['alpha']):>6}{d['n']:>5}{d['r2']:>8.3f}   {d['eq']}")

# paired delta-c at fixed geometry + pooled alpha-regressor test (>= 2 configs)
keys = [f for f in mc_files if isinstance(loaded[f]["alpha"], float)]
if len(keys) >= 2:
    a, b = sorted(keys, key=lambda f: loaded[f]["alpha"])[:2]
    shared = sorted(set(loaded[a]["runs"]) & set(loaded[b]["runs"]))
    dC = np.array([loaded[b]["runs"][k] - loaded[a]["runs"][k] for k in shared])
    Hp = np.concatenate([loaded[f]["H"] for f in keys])
    Cp = np.concatenate([loaded[f]["C"] for f in keys])
    Ap = np.concatenate([np.full(loaded[f]["n"], loaded[f]["alpha"]) for f in keys])
    def r2(X, y):
        p = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    r2_h = r2(np.column_stack([np.ones_like(Hp), Hp]), Cp)
    r2_ha = r2(np.column_stack([np.ones_like(Hp), Hp, Ap]), Cp)
    print(f"\npaired delta_c (alpha {loaded[a]['alpha']}->{loaded[b]['alpha']}, n={len(shared)}): "
          f"mean={dC.mean():+.2f} std={dC.std():.2f}")
    print(f"pooled (n={len(Hp)}): c~H R2={r2_h:.3f} | c~H+alpha R2={r2_ha:.3f} | dR2={r2_ha - r2_h:+.3f}")

# --------------------------------------------------------------------- FIGURE
main = loaded[MAIN_CONFIG]
fig, ax = plt.subplots(figsize=(9, 7))
sc = ax.scatter(main["H"], main["C"], c=main["RMSE"], cmap="viridis", alpha=0.6, s=60)
plt.colorbar(sc, ax=ax, label="Min RMSE")

Hr = np.linspace(main["H"].min(), main["H"].max(), 100)
colors = plt.cm.plasma(np.linspace(0, 0.85, len(mc_files)))
for f, col in zip(mc_files, colors):
    d = loaded[f]
    is_main = (f == MAIN_CONFIG)
    ax.plot(Hr, d["fn"](Hr),
            color="red" if is_main else col,
            lw=3 if is_main else 1.8,
            ls="-" if is_main else "--",
            label=f"α={d['alpha']}: c={d['eq'].split('=', 1)[1].strip()} (R²={d['r2']:.3f})")

ax.set_xlabel("Normalized H_src (H_src / V^(1/3))", fontsize=11)
ax.set_ylabel("Optimal c", fontsize=11)
ax.set_title(f"c vs H_src across absorption ({MODEL} fits; scatter = α={main['alpha']})", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="best")
plt.tight_layout()

if SAVE:
    plt.savefig(OUT, dpi=150)
    print(f"\nSaved {OUT}")
plt.show()
