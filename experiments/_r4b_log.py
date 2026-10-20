# -*- coding: utf-8 -*-
"""Archive R4b/R4c and append results (numerals from CSVs; claims asserted).

    python experiments/_r4b_log.py
"""
import sys, os, csv, json, shutil, hashlib
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\zhibo.zhu\OneDrive - UIH Group\Documents\Jupyter\diffusion"
os.chdir(ROOT)
MD = "WEEKEND_ITERATIONS.md"
HEAD = "### R4b / R4c results"
assert HEAD not in open(MD, encoding="utf-8").read()


def curve(run):
    rows = list(csv.DictReader(open(os.path.join(run, "train_val.csv"), encoding="utf-8")))
    g = [(int(r["iter"]), float(r["val_neff_gmean"])) for r in rows if int(r["iter"]) % 200 == 0]
    return max(g, key=lambda z: z[1]), g[-1]


runs = {"R4": "runs/r4_subset/l1", "R4b": "runs/r4b_subset/bal/l1", "R4c": "runs/r4b_subset/bal_aug/l1"}
cur = {k: curve(v) for k, v in runs.items()}
sel = max(cur, key=lambda k: cur[k][0][1])
assert sel == "R4c"


def load_sel(path):
    S = list(csv.DictReader(open(path, encoding="utf-8")))
    return {(r["region"], int(float(r["a"])), int(float(r["slice"]))): r for r in S}


A = load_sel("runs/r4_subset/l1/eval/split_test/split_selected.csv")
C = load_sel("runs/r4b_subset/bal_aug/l1/eval/split_test/split_selected.csv")
test = (25, 22, 8)
f = lambda d, k, col: float(d[k][col])
gain = {s: f(C, ("signal", 9, s), "r4") / f(A, ("signal", 9, s), "r4") for s in test}
vs_r3 = {s: f(C, ("signal", 9, s), "r4") / f(C, ("signal", 9, s), "r3avg") for s in test}
assert all(f(C, ("signal", a, s), "r4") > max(f(C, ("signal", a, s), "vanilla"), f(C, ("signal", a, s), "blur"))
           for a in (1, 5, 9) for s in test)
assert vs_r3[25] < 1 and vs_r3[22] < 1 and vs_r3[8] > 1

ck = "runs/r4b_subset/bal_aug/l1/ckpt/l1_%06d.pkl" % cur["R4c"][0][0]
cd = "checkpoints/R4_SUBSET"
dst = os.path.join(cd, "r4c_" + os.path.basename(ck))
shutil.copyfile(ck, dst)
h = hashlib.sha256(open(dst, "rb").read()).hexdigest()
man = os.path.join(cd, "MANIFEST.md")
if h not in open(man, encoding="utf-8").read():
    open(man, "a", encoding="utf-8").write("| l1 (R4c bal+aug) | %d | `%s` | %.1f MB | `%s` |\n" % (
        cur["R4c"][0][0], dst.replace("\\", "/"), os.path.getsize(dst) / 1048576, h))
for src, slug in (("runs/r4b_subset/stdout_bal.txt", "train_r4b_bal"),
                  ("runs/r4b_subset/stdout_bal_aug.txt", "train_r4c_bal_aug"),
                  ("runs/r4b_subset/bal_aug/l1/split_stdout.txt", "split_r4c")):
    t = open(src, encoding="utf-8", errors="replace").read()
    e = t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    open("Archive/R4_SUBSET/stdout_%s_raw.html" % slug, "w", encoding="utf-8").write(
        "<!doctype html><meta charset=\"utf-8\"><title>%s</title><h1>RUN 2026-10-20 &mdash; %s</h1>"
        "<p>Verbatim stdout, unedited.</p><pre>%s</pre>\n" % (slug, slug, e))

L = ["", HEAD, "",
     "| variant | val n_eff gmean at best saved ckpt | best iter | at iter 3000 |", "|---|---:|---:|---:|"]
for k in ("R4", "R4b", "R4c"):
    (bi, bg), (li, lg) = cur[k]
    L.append("| %s | %.2f | %d | %.2f |" % (k, bg, bi, lg))
L += ["",
      "- Balancing alone (R4b) does not beat R4 on val; balancing + dihedral augmentation (R4c) does and, "
      "unlike R4/R4b, has not overfit by 3000 iterations. R4c (iter %d) is selected by the declared rule."
      % cur["R4c"][0][0],
      "- Split test, signal region, a=9, test slices — R4c / R4: %s. The coverage fix lands where it was "
      "aimed (the quiet slices)." % ", ".join("s%d %.2fx" % (s, gain[s]) for s in test),
      "- R4c beats plain averaging and the val-tuned blur on every test slice at a = 1, 5, 9.",
      "- R4c / R3 at a=9: %s. R4c still trails R3 on the two quiet test slices and leads on the noisy one; "
      "with 28 training slices, one model covering every averaging depth appears data-limited. R3 stays the "
      "basis for the proof." % ", ".join("s%d %.2f" % (s, vs_r3[s]) for s in test),
      "",
      "**Artifacts:** `runs/r4b_subset/{bal,bal_aug}/l1/` (ckpt, `train_val.csv`, `eval/curves.png`), "
      "`runs/r4b_subset/bal_aug/l1/eval/split_test/` (same figures/maps as R4). Checkpoint copy: "
      "`checkpoints/R4_SUBSET/MANIFEST.md` (`%s`). Stdout: `Archive/R4_SUBSET/stdout_{train_r4b_bal,"
      "train_r4c_bal_aug,split_r4c}_raw.html`. Commands: `python experiments/_r4b_train.py --tag bal` / "
      "`--tag bal_aug --aug`; `python experiments/_r4_split_test.py --r4-run runs/r4b_subset/bal_aug/l1 "
      "--r4-iter %d`." % (h[:16], cur["R4c"][0][0]), ""]
open(MD, "a", encoding="utf-8").write("\n".join(L))
print("\n".join(L))
