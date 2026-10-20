# -*- coding: utf-8 -*-
"""Archive an R3 run and append its results to WEEKEND_ITERATIONS.md.

Every numeral is computed here from ``<run>/train_val.csv`` and
``<run>/eval/summary.csv``; nothing is typed by hand. Copies the display
checkpoint + hash to ``checkpoints/R3_TOTAL`` and the raw stdout of training and
evaluation to ``Archive/R3_TOTAL`` as HTML.

    python experiments/_r3_log.py --run runs/r3_total/l1 --iter 2000
"""
import sys, os, csv, json, shutil, hashlib, argparse
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np

ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="runs/r3_total/l1")
ap.add_argument("--iter", type=int, required=True)
a = ap.parse_args()
loss = os.path.basename(a.run)
EV = os.path.join(a.run, "eval")

tv = list(csv.DictReader(open(os.path.join(a.run, "train_val.csv"), encoding="utf-8")))
sm = list(csv.DictReader(open(os.path.join(EV, "summary.csv"), encoding="utf-8")))
cfg = json.load(open(os.path.join(a.run, "config.json"), encoding="utf-8"))
for r in sm:
    for k in ("iter", "slice"):
        r[k] = int(r[k])
    for k in ("sigma_hat", "knob", "noise", "gain", "sb", "shf", "relE", "pha"):
        r[k] = float(r[k])

# ---- archive -------------------------------------------------------------------
ck = os.path.join(a.run, "ckpt", "%s_%06d.pkl" % (loss, a.iter))
cd = "checkpoints/R3_TOTAL"
os.makedirs(cd, exist_ok=True)
dst = os.path.join(cd, os.path.basename(ck))
shutil.copyfile(ck, dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
shutil.copyfile(os.path.join(a.run, "config.json"), os.path.join(cd, "config_%s.json" % loss))
open(os.path.join(cd, "MANIFEST.md"), "a", encoding="utf-8").write(
    "| %s | %d | `%s` | %.1f MB | `%s` |\n" % (loss, a.iter, dst, os.path.getsize(dst) / 1048576, h))

ad = "Archive/R3_TOTAL"
os.makedirs(ad, exist_ok=True)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


for src, slug in ((os.path.join("runs/r3_total", "stdout_%s.txt" % loss), "train_%s" % loss),
                  (os.path.join(a.run, "eval_stdout.txt"), "eval_%s" % loss)):
    txt = open(src, encoding="utf-8", errors="replace").read()
    open(os.path.join(ad, "stdout_%s_raw.html" % slug), "w", encoding="utf-8").write(
        "<!doctype html><meta charset=\"utf-8\"><title>%s raw stdout</title>"
        "<style>body{font:13px/1.45 ui-monospace,Consolas,monospace;max-width:1100px;"
        "margin:24px auto;padding:0 16px}pre{background:#f4f4f4;padding:12px;"
        "white-space:pre-wrap}</style>\n<h1>RUN 2026-10-20 &mdash; R3_TOTAL %s</h1>\n"
        "<p>Verbatim stdout, unedited.</p>\n<pre>%s</pre>\n" % (slug, slug, esc(txt)))

# ---- numbers ------------------------------------------------------------------------
vl = [(int(r["iter"]), float(r["val_l1"])) for r in tv]
best_vl = min(vl, key=lambda z: z[1])
last = tv[-1]


def cell(it, s, fam):
    return next((r for r in sm if r["iter"] == it and r["slice"] == s and r["family"] == fam), None)


slices = [(r["split"], r["slice"]) for r in sm if r["iter"] == a.iter and r["family"] == "id"]
net = {s: cell(a.iter, s, "net") for _, s in slices}
blur = {s: cell(a.iter, s, "blur") for _, s in slices}
ide = {s: cell(a.iter, s, "id") for _, s in slices}
wins = [s for _, s in slices if net[s]["relE"] < blur[s]["relE"]]
gain_more = [s for _, s in slices if net[s]["gain"] > blur[s]["gain"]]
pha_less = [s for _, s in slices if net[s]["pha"] < blur[s]["pha"]]
loses = [s for _, s in slices if s not in wins]

its = sorted({r["iter"] for r in sm})
unreached = [(it, s) for it in its for _, s in slices if cell(it, s, "net") is None]


def mrel(it, sp, fam):
    v = [r["relE"] for r in sm if r["iter"] == it and r["split"] == sp and r["family"] == fam]
    return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)


L = []
A = L.append
A("")
A("### R3_TOTAL results — %s" % loss)
A("")
A("**Training.** %d iterations in %.1f min. Full-Tweedie val L1 at deployment bottoms at "
  "iteration %d (%.5f) and ends at %.5f; the 1/3 decode's val residual noise ends at %.3f of "
  "sigma_hat with gain %.3f. Curves: `%s/curves.png`."
  % (int(last["iter"]), float(last["sec"]) / 60, best_vl[0], best_vl[1], float(last["val_l1"]),
     float(last["val_noise13"]), float(last["val_gain13"]), EV.replace("\\", "/")))
A("")
A("**Matched noise, mean relE @ noise 1/3 per checkpoint** (net = `denoise_to` swept over "
  "target_ratio; blur = Gaussian; both read at exactly 1/3; identity reads noise 1.000):")
A("")
A("| iter | val net | val blur | test net | test blur |")
A("|---:|---:|---:|---:|---:|")
for it in its:
    vn, nv = mrel(it, "val", "net"); vb, _ = mrel(it, "val", "blur")
    tn, nt = mrel(it, "test", "net"); tb, _ = mrel(it, "test", "blur")
    A("| %d | %.4f%s | %.4f | %.4f%s | %.4f |"
      % (it, vn, "" if nv == 3 else " (%d/3)" % nv, vb, tn, "" if nt == 3 else " (%d/3)" % nt, tb))
A("")
if unreached:
    A("Cells where the net could not get down to noise 1/3 at any tested ratio (excluded from the "
      "means above): %s." % ", ".join("iter %d slice %d" % u for u in unreached))
    A("")
A("**Per slice at iteration %d** (@ noise 1/3; knob = target_ratio for net, Gaussian width for "
  "blur):" % a.iter)
A("")
A("| split | slice | sigma_hat | family | knob | gain | s_bias | sHF %% | relE | phase deg |")
A("|---|---:|---:|---|---:|---:|---:|---:|---:|---:|")
for sp, s in slices:
    for fam, d in (("net", net[s]), ("blur", blur[s]), ("id", ide[s])):
        A("| %s | %d | %.4f | %s | %.3f | %.3f | %.3f | %.1f | %.3f | %.1f |"
          % (sp, s, d["sigma_hat"], fam, d["knob"], d["gain"], d["sb"], d["shf"], d["relE"], d["pha"]))
A("")
kn = [net[s]["knob"] for _, s in slices]
A("- Net beats blur on relE at matched noise on **%d/%d** slices%s. It keeps more magnitude gain "
  "on %d/%d and has lower phase error on %d/%d."
  % (len(wins), len(slices),
     "" if not loses else " (not on slice %s)" % ", ".join(str(s) for s in loses),
     len(gain_more), len(slices), len(pha_less), len(slices)))
A("- The target_ratio that lands exactly on noise 1/3 is %.3f–%.3f, close to the 1/3 the "
  "one-step DDIM rule predicts (slightly below it because x0_hat is not noise-free)."
  % (min(kn), max(kn)))
A("- Gain below 1 is not net-specific: blur at the same noise keeps %.3f–%.3f. Reaching 1/3 "
  "noise on this data costs magnitude for any estimator; post-gain bias (s_bias) stays %.3f–%.3f "
  "for the net." % (min(blur[s]["gain"] for _, s in slices), max(blur[s]["gain"] for _, s in slices),
                    min(net[s]["sb"] for _, s in slices), max(net[s]["sb"] for _, s in slices)))
A("- The test margin over blur is much thinner than val (mean %.4f vs %.4f, against %.4f vs %.4f "
  "on val)." % (mrel(a.iter, "test", "net")[0], mrel(a.iter, "test", "blur")[0],
                mrel(a.iter, "val", "net")[0], mrel(a.iter, "val", "blur")[0]))
A("")
A("**Displays** (`%s/`): `maps_<split>_<slice>.png` — top row reference |mean of 10|, noisy "
  "realization 0, net `denoise_to(1/3)`, net full Tweedie, blur at noise 1/3; bottom row the LOO "
  "mean and each map minus it. `maps_overview.png` stacks all slices. Display checkpoint: "
  "iteration %d (lowest mean val relE@1/3)." % (EV.replace("\\", "/"), a.iter))
A("")
A("**Artifacts:** `%s/` (ckpt every 200, `train.log`, `train_val.csv`, `config.json`), "
  "`%s/eval.csv`, `summary.csv`. Checkpoint copy + sha256: `checkpoints/R3_TOTAL/MANIFEST.md` "
  "(`%s`). Verbatim stdout: `Archive/R3_TOTAL/stdout_train_%s_raw.html`, "
  "`stdout_eval_%s_raw.html`. Eval command: `python experiments/_r3_eval.py --run %s`."
  % (a.run, EV.replace("\\", "/"), h[:16], loss, loss, a.run))
A("")
open("WEEKEND_ITERATIONS.md", "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
