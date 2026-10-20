# -*- coding: utf-8 -*-
"""CV_ABLATION report: {L1, MSE} x {unweighted, noise-normalized}, paired over 34 slices.

Implements ``Archive/CV_ABLATION/PREREGISTRATION.md``. Arm A = CV_R3 (read only);
arms B/C/D from ``runs/cv_ablation/<arm>/eval``. Every numeral is computed here
from the per-slice CSVs; every verdict is computed from the pre-registered rule
(nothing about the outcome is written in advance). Sanity asserts: the plain
and oracle-blur numbers must be identical across arms (only the model changed).

Outputs: ``runs/cv_ablation/report/`` (paired.csv, summary.json, figures), an
appended WEEKEND_ITERATIONS.md section (once), ``checkpoints/CV_ABLATION/
MANIFEST.md`` (sha256 of the evaluated weights, hashed in place) and raw stdout
as HTML in ``Archive/CV_ABLATION``.

    python experiments/_cv_ablation_report.py
"""
import sys, os, csv, json, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import wilcoxon, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
OUT = "runs/cv_ablation/report"
MD = "WEEKEND_ITERATIONS.md"
HEAD = "## RUN 2026-10-20 — CV_ABLATION results"
ARMS = {"A": ("runs/cv_r3", "L1, unweighted (CV_R3)"),
        "B": ("runs/cv_ablation/mse", "MSE, unweighted"),
        "C": ("runs/cv_ablation/mse_norm", "MSE, noise-normalized"),
        "D": ("runs/cv_ablation/l1_norm", "L1, noise-normalized")}
A_SIZES = (1, 2, 3, 5, 7, 9)
REGIONS = ("signal", "all")
NBOOT = 10000
os.makedirs(OUT, exist_ok=True)


def load(arm):
    d = ARMS[arm][0]
    P = {}
    for r in csv.DictReader(open(os.path.join(d, "eval", "per_slice.csv"), encoding="utf-8")):
        P[(int(r["slice"]), int(r["a"]), r["region"])] = {k: float(v) for k, v in r.items()
                                                         if k not in ("region",)}
    B = {int(r["slice"]): {k: float(v) for k, v in r.items()}
         for r in csv.DictReader(open(os.path.join(d, "eval", "bands.csv"), encoding="utf-8"))}
    return P, B


data = {k: load(k) for k in ARMS}
slices = sorted({k[0] for k in data["A"][0]})
assert len(slices) == 34
sig = {s: data["A"][0][(s, 1, "signal")]["sigma_hat"] for s in slices}

# sanity: only the model changed between arms
for arm in "BCD":
    for key, ra in data["A"][0].items():
        rb = data[arm][0][key]
        assert np.isclose(ra["plain"], rb["plain"], rtol=1e-9), (arm, key)
        assert np.isclose(ra["blur_oracle"], rb["blur_oracle"], rtol=1e-9), (arm, key)

rng = np.random.default_rng(0)


def paired(num, den, a, region):
    lr = np.log(np.array([data[num][0][(s, a, region)]["r3"] / data[den][0][(s, a, region)]["r3"]
                          for s in slices]))
    boot = rng.choice(lr, size=(NBOOT, len(lr))).mean(1)
    p = wilcoxon(lr, alternative="two-sided").pvalue if np.any(lr != 0) else 1.0
    return dict(num=num, den=den, a=a, region=region, n=len(lr), gmean=float(np.exp(lr.mean())),
                ci_lo=float(np.exp(np.percentile(boot, 2.5))), ci_hi=float(np.exp(np.percentile(boot, 97.5))),
                wins=int((lr > 0).sum()), p=float(p), lr=lr)


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


# ---- primary ----------------------------------------------------------------------------
prim = {arm: paired(arm, "A", 5, "signal") for arm in "BCD"}
adj = holm([prim[a]["p"] for a in "BCD"])
for arm, q in zip("BCD", adj):
    prim[arm]["p_holm"] = float(q)
    prim[arm]["verdict"] = ("better than A" if q < 0.05 and prim[arm]["gmean"] > 1 else
                            "worse than A" if q < 0.05 and prim[arm]["gmean"] < 1 else
                            "no significant difference")

# ---- secondary: every a, both regions --------------------------------------------------
sec = [paired(arm, "A", a, reg) for reg in REGIONS for a in A_SIZES for arm in "BCD"]
with open(os.path.join(OUT, "paired.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=[k for k in sec[0] if k != "lr"])
    w.writeheader()
    for r in sec:
        w.writerow({k: v for k, v in r.items() if k != "lr"})


def vs_baselines(arm, a=5, region="signal"):
    out = {}
    for base in ("plain", "blur_oracle"):
        lr = np.log(np.array([data[arm][0][(s, a, region)]["r3"] / data[arm][0][(s, a, region)][base]
                              for s in slices]))
        out[base] = (float(np.exp(lr.mean())), int((lr > 0).sum()))
    return out


base5 = {arm: vs_baselines(arm) for arm in ARMS}
blur_sig = {arm: [a for a in A_SIZES
                  if wilcoxon(np.log([data[arm][0][(s, a, "signal")]["r3"] /
                                      data[arm][0][(s, a, "signal")]["blur_oracle"] for s in slices]),
                              alternative="greater").pvalue < 0.01] for arm in ARMS}

# ---- hypotheses --------------------------------------------------------------------------
h1 = prim["B"]
rho_c = spearmanr([sig[s] for s in slices], paired("C", "B", 5, "signal")["lr"])
rho_d = spearmanr([sig[s] for s in slices], paired("D", "A", 5, "signal")["lr"])
mse_vs_l1_norm = paired("C", "D", 5, "signal")

# ---- bands, slice 29 --------------------------------------------------------------------------
band = {}
for arm in "BCD":
    band[arm] = []
    for b in range(4):
        r = np.array([data[arm][1][s][f"r3_band{b}"] / data["A"][1][s][f"r3_band{b}"] for s in slices])
        band[arm].append((float(np.median(r)), int((r > 1).sum())))
s29 = {arm: {a: (data[arm][0][(29, a, "signal")]["r3"], data[arm][0][(29, a, "signal")]["blur_oracle"])
             for a in (1, 5, 9)} for arm in ARMS}

# ---- figures ------------------------------------------------------------------------------
gm = lambda arm, a, key="r3": float(np.exp(np.mean(np.log([data[arm][0][(s, a, "signal")][key] for s in slices]))))
fig, ax = plt.subplots(figsize=(6.8, 4.6))
for arm, c in zip("ABCD", ("C3", "C1", "C2", "C4")):
    ax.plot(A_SIZES, [gm(arm, a) for a in A_SIZES], "o-", color=c, label=f"{arm}: {ARMS[arm][1]}")
ax.plot(A_SIZES, [gm("A", a, "blur_oracle") for a in A_SIZES], "d--", color="C0", label="oracle blur")
ax.plot(A_SIZES, A_SIZES, "ks:", label="plain average")
ax.set_yscale("log"); ax.set_xticks(A_SIZES); ax.set_xlabel("realizations used (a)")
ax.set_ylabel("geometric-mean n_eff over 34 slices (log)"); ax.legend(fontsize=7)
ax.set_title("Loss ablation, signal region, held-out models", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "abl_neff_vs_a.png"), dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, arm in zip(axs, "BCD"):
    lr = prim[arm]["lr"]
    ax.scatter([sig[s] for s in slices], np.exp(lr), s=18, color="C3")
    for s, v in zip(slices, np.exp(lr)):
        if s == 29:
            ax.annotate("29", (sig[s], v), fontsize=7)
    ax.axhline(1, color="k", lw=0.8); ax.set_yscale("log")
    ax.set_title(f"{arm} ({ARMS[arm][1]}) / A, a=5", fontsize=9); ax.set_xlabel("slice sigma_hat")
axs[0].set_ylabel("per-slice n_eff ratio vs A (log)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "abl_paired_ratio_vs_sigma.png"), dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 4))
for i, (arm, c) in enumerate(zip("BCD", ("C1", "C2", "C4"))):
    ax.bar(np.arange(4) + (i - 1) * 0.25, [m for m, _ in band[arm]], width=0.25, color=c,
           label=f"{arm}: {ARMS[arm][1]}")
ax.axhline(1, color="k", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels(["0-0.0625", "0.0625-0.125", "0.125-0.25", "0.25-0.75"])
ax.set_xlabel("frequency band (cycles/pixel)"); ax.set_ylabel("median per-band n_eff ratio vs A")
ax.set_title("Where in frequency each loss gains or loses (a = 5)", fontsize=9); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "abl_bands_a5.png"), dpi=110); plt.close(fig)

# ---- summary for the argument document -------------------------------------------------------
summary = dict(prim={a: {k: v for k, v in prim[a].items() if k != "lr"} for a in "BCD"},
               base5=base5, blur_sig=blur_sig, h1=dict(gmean=h1["gmean"], p_holm=h1["p_holm"]),
               rho_c=[float(rho_c.correlation), float(rho_c.pvalue)],
               rho_d=[float(rho_d.correlation), float(rho_d.pvalue)],
               c_vs_d={k: v for k, v in mse_vs_l1_norm.items() if k != "lr"},
               band=band, s29=s29, labels={k: v[1] for k, v in ARMS.items()})
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)

# ---- archive ------------------------------------------------------------------------------------
os.makedirs("checkpoints/CV_ABLATION", exist_ok=True)
man = ["# CV_ABLATION checkpoint manifest", "",
       "Evaluated weights (iteration 2000) of the ablation arms, hashed IN PLACE (not copied, to avoid",
       "duplicating 15 x 25 MB on OneDrive). Arm A is in `checkpoints/CV_R3/MANIFEST.md`.", "",
       "| arm | fold | path | sha256 |", "|---|---:|---|---|"]
for arm in "BCD":
    d = ARMS[arm][0]
    for k in range(5):
        cfg = json.load(open(os.path.join(d, f"fold{k}", "config.json"), encoding="utf-8"))
        p = os.path.join(d, f"fold{k}", "ckpt", "%s_002000.pkl" % cfg["ckpt_cfg"]["loss"])
        man.append("| %s | %d | `%s` | `%s` |" % (arm, k, p.replace("\\", "/"),
                                                  hashlib.sha256(open(p, "rb").read()).hexdigest()))
open("checkpoints/CV_ABLATION/MANIFEST.md", "w", encoding="utf-8").write("\n".join(man) + "\n")
for f in sorted(os.listdir("runs/cv_ablation")):
    if f.startswith(("stdout_", "eval_stdout_")) and f.endswith(".txt"):
        t = open(os.path.join("runs/cv_ablation", f), encoding="utf-8", errors="replace").read()
        e = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        open(os.path.join("Archive/CV_ABLATION", f[:-4] + "_raw.html"), "w", encoding="utf-8").write(
            "<!doctype html><meta charset=\"utf-8\"><title>%s</title><h1>CV_ABLATION %s</h1>"
            "<p>Verbatim stdout, unedited.</p><pre>%s</pre>\n" % (f, f, e))

# ---- log section ---------------------------------------------------------------------------------
fmtp = lambda r: "%.3f× (95%% CI %.3f–%.3f; higher on %d/%d slices; p = %.2g, Holm %.2g)" % (
    r["gmean"], r["ci_lo"], r["ci_hi"], r["wins"], r["n"], r["p"], r["p_holm"])
L = ["", "---", "", HEAD, "",
     "Pre-registered (`Archive/CV_ABLATION/PREREGISTRATION.md`). Paired over the 34 held-out slices; "
     "plain and oracle-blur numbers verified identical across arms (only the model changed).", "",
     "**PRIMARY (signal, a = 5, n_eff ratio vs A = L1 unweighted):**", ""]
for arm in "BCD":
    L.append("- %s (%s): %s → **%s**" % (arm, ARMS[arm][1], fmtp(prim[arm]), prim[arm]["verdict"]))
L += ["", "**Each arm against the baselines (signal, a = 5; geometric mean, slices won):**", "",
      "| arm | vs plain averaging | vs oracle blur | a where significantly better than oracle blur (p<0.01) |",
      "|---|---|---|---|"]
for arm in ARMS:
    L.append("| %s %s | %.2f× (%d/34) | %.3f× (%d/34) | %s |" % (
        arm, ARMS[arm][1], base5[arm]["plain"][0], base5[arm]["plain"][1],
        base5[arm]["blur_oracle"][0], base5[arm]["blur_oracle"][1], ", ".join(map(str, blur_sig[arm])) or "none"))
L += ["", "**Secondary — ratio vs A at every a (signal; unadjusted p):**", "",
      "| a | B MSE | C MSE norm | D L1 norm |", "|---:|---|---|---|"]
for a in A_SIZES:
    cells = []
    for arm in "BCD":
        r = next(q for q in sec if q["num"] == arm and q["a"] == a and q["region"] == "signal")
        cells.append("%.3f× [%.3f–%.3f], %d/34, p=%.2g" % (r["gmean"], r["ci_lo"], r["ci_hi"], r["wins"], r["p"]))
    L.append("| %d | %s |" % (a, " | ".join(cells)))
L += ["",
      "- **H1 (MSE ≥ L1):** B/A = %.3f×, Holm p = %.2g → %s." % (
          h1["gmean"], h1["p_holm"],
          "supported" if h1["gmean"] > 1 and h1["p_holm"] < 0.05 else
          "contradicted (MSE significantly worse)" if h1["gmean"] < 1 and h1["p_holm"] < 0.05 else
          "not supported (no significant difference)"),
      "- **H2 (normalization helps quiet slices most):** Spearman of per-slice ratio vs σ̂ — C/B: "
      "rho = %.2f (p = %.2g); D/A: rho = %.2f (p = %.2g) → %s." % (
          rho_c.correlation, rho_c.pvalue, rho_d.correlation, rho_d.pvalue,
          "supported" if (rho_c.correlation < 0 and rho_c.pvalue < 0.05) and
                         (rho_d.correlation < 0 and rho_d.pvalue < 0.05) else
          "partly supported" if (rho_c.correlation < 0 and rho_c.pvalue < 0.05) or
                                (rho_d.correlation < 0 and rho_d.pvalue < 0.05) else "not supported"),
      "- C vs D (MSE vs L1, both normalized): %.3f× [%.3f–%.3f], %d/34, p = %.2g." % (
          mse_vs_l1_norm["gmean"], mse_vs_l1_norm["ci_lo"], mse_vs_l1_norm["ci_hi"],
          mse_vs_l1_norm["wins"], mse_vs_l1_norm["p"]),
      "- Per frequency band (a = 5, median ratio vs A; slices above 1): " + "; ".join(
          "%s: %s" % (arm, ", ".join("%.2f (%d)" % mb for mb in band[arm])) for arm in "BCD") + ".",
      "- Slice 29 (out-of-range quiet slice), n_eff vs oracle blur at a = 1 / 5 / 9: " + "; ".join(
          "%s %s" % (arm, " / ".join("%.1f vs %.1f" % s29[arm][a] for a in (1, 5, 9))) for arm in ARMS) + ".",
      "- Caveat (pre-registered): the best of four arms on the same 34 slices is mildly optimistic; it is "
      "the best of this ablation, not a second confirmed claim. CV_R3 (arm A) stays the headline.", "",
      "**Figures:** `%s/abl_neff_vs_a.png`, `abl_paired_ratio_vs_sigma.png`, `abl_bands_a5.png`; per-arm "
      "maps and figures under `runs/cv_ablation/<arm>/eval/`. **Data:** `%s/paired.csv`, `summary.json`. "
      "**Commands:** `python experiments/_cv_train.py --fold K --loss {mse,l1} [--noise-norm] --outdir "
      "runs/cv_ablation/<arm>`; `python experiments/_cv_eval.py --cv runs/cv_ablation/<arm>`; "
      "`python experiments/_cv_ablation_report.py`. Weights: `checkpoints/CV_ABLATION/MANIFEST.md`. "
      "Stdout: `Archive/CV_ABLATION/*_raw.html`." % (OUT, OUT), ""]
if HEAD not in open(MD, encoding="utf-8").read():
    open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
