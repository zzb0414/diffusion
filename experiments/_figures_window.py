# -*- coding: utf-8 -*-
"""Display window shared by ``_figures.py`` (Figs. 1-2) and ``_abstract.py`` (their captions).

Grey window per slice = [0, WIN x 99.5th percentile of the plain mean of 10], the
same for every panel of that slice. WIN opens the window slightly beyond that
percentile (display only) so the noisy repetition is not dominated by saturation
while the denoised maps keep their contrast.
"""
WIN = 1.3
