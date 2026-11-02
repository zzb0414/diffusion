# -*- coding: utf-8 -*-
"""C7 theory leg (exploratory): in the linear-Gaussian model with infinite data, how much does "denoise each copy
(Wiener gain w1 = S/(S+s2)) + best dial r, then average" lose against "Wiener on the copy mean" (w_a = S/(S+s2/a),
the MMSE estimate since the mean is sufficient)? Same spectrum as _theory_c2_toy.py; n_eff = s2 / MSE.

    python experiments/_theory_c7_gaussian.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
H = 64
f = np.hypot(np.fft.fftfreq(H)[:, None], np.fft.fftfreq(H)[None, :])
S = 1.0 / (f ** 2 + 0.02 ** 2); S /= S.mean()
mse = lambda w, s2, a: float(((1 - w) ** 2 * S).mean() + (w ** 2).mean() * s2 / a)
print("per-pixel SNR | a | n_eff per-copy Wiener + best r | n_eff Wiener on the mean | ratio (mean / per-copy)")
for s2 in (0.04, 1.0, 25.0, 400.0):
    for a in (1, 3, 9):
        w1 = S / (S + s2)
        best = max(s2 / mse(w1 + r * (1 - w1), s2, a) for r in np.linspace(0, 1, 201))
        wa = S / (S + s2 / a)
        nm = s2 / mse(wa, s2, a)
        print("  %6.3f      | %d | %8.2f                       | %8.2f                 | %.3f" % (1 / np.sqrt(s2), a, best, nm, nm / best))
