"""The glance lnL / GB-leaf figures offer a longer recent window (2026-11-04).

User request: in a long full_pe the 50-row recent-window panel is too short;
show the last 200 rows, switchable back to 50 -- and never a "last 200" over a
run that has fewer rows than that. ``history_windows`` adds a long window only
while it is still at most half the stored rows; ``fig_history_set`` renders one
figure per window and ``_window_img`` emits a CSS-only toggle between them.
"""
import re
import unittest
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402

from lisatools.globalfit.monitor import _short as sh  # noqa: E402
from lisatools.globalfit.monitor._short import _generator_functions  # noqa: E402


def compile_ns():
    ns = {"np": np}
    _generator_functions(ns, names=("walker_color", "plot_walker_lines", "mark_stages",
                                    "_zoom_its"),
                         consts=("WALKER_COLORS", "WALKER_OTHER_COLOR", "WALKER_MEAN_COLOR"))
    ns["STAGE_BOUNDS"] = []
    return ns


NS = compile_ns()


class HistoryWindowsTest(unittest.TestCase):
    def test_tiers(self):
        want = {0: [1], 3: [3], 14: [5], 40: [10], 99: [25], 120: [50], 160: [50],
                199: [50], 200: [100, 50], 399: [100, 50], 400: [200, 50], 5000: [200, 50]}
        for n, w in want.items():
            with self.subTest(n=n):
                self.assertEqual(sh.history_windows(NS, n), w)

    def test_never_wider_than_half_the_run_or_the_rows(self):
        for n in range(1, 1200):
            w = sh.history_windows(NS, n)
            self.assertLessEqual(max(w), n)
            self.assertEqual(w, sorted(set(w), reverse=True))
            if len(w) > 1:
                self.assertLessEqual(w[0], n // 2)


def _figures(y):
    """``[(nz, Figure), ...]`` from fig_history_set, PNG encoding intercepted."""
    got = []
    with mock.patch.object(sh, "_png", lambda fig, dpi=None: got.append(fig) or f"png{len(got)}"):
        out = sh.fig_history_set(NS, y, "lnL", "cold log-likelihood")
    return [(nz, got[int(p[3:]) - 1]) for nz, p in out]


class FigureSetTest(unittest.TestCase):
    def _check(self, nit, want):
        y = np.cumsum(np.random.default_rng(0).normal(size=(nit, 4)), axis=0)
        figs = _figures(y)
        self.assertEqual([nz for nz, _ in figs], want)
        for nz, fig in figs:
            ax1 = fig.axes[1]
            self.assertEqual(ax1.get_title(), f"last {nz} rows")
            for ln in ax1.get_lines()[:4]:
                self.assertEqual(len(ln.get_xdata()), nz)
            lo, hi = ax1.get_xlim()
            self.assertAlmostEqual(lo, nit - nz - 0.5)
            self.assertAlmostEqual(hi, nit - 0.5)

    def test_short_run_draws_one_window(self):
        self._check(160, [50])

    def test_mid_run_draws_100_and_50(self):
        self._check(250, [100, 50])

    def test_long_run_draws_200_and_50(self):
        self._check(450, [200, 50])

    def test_explicit_window_is_clamped_to_rows(self):
        got = []
        y = np.zeros((30, 4))
        with mock.patch.object(sh, "_png", lambda fig, dpi=None: got.append(fig) or "p"):
            sh.fig_history(NS, y, "lnL", "t", nz=200)
        self.assertEqual(got[0].axes[1].get_title(), "last 30 rows")


class ToggleHtmlTest(unittest.TestCase):
    def test_single_variant_is_a_plain_img(self):
        h = sh._window_img([(50, "AAA")], "cold log-likelihood", "lnl")
        self.assertEqual(h, sh._img("AAA", "cold log-likelihood"))
        self.assertNotIn("radio", h)

    def test_missing_is_the_missing_box(self):
        self.assertIn('class="missing"', sh._window_img(None, "GB leaf count", "leaves"))
        self.assertIn('class="missing"', sh._window_img([], "GB leaf count", "leaves"))

    def test_two_variants_long_checked_first(self):
        h = sh._window_img([(200, "LONG"), (50, "SHORT")], "GB leaf count", "leaves")
        radios = re.findall(r'<input type="radio"[^>]*>', h)
        self.assertEqual(len(radios), 2)
        self.assertIn('id="win-leaves-200"', radios[0])
        self.assertIn("checked", radios[0])
        self.assertNotIn("checked", radios[1])
        self.assertIn('<label for="win-leaves-200">last 200 rows</label>', h)
        self.assertIn('<label for="win-leaves-50">last 50 rows</label>', h)
        # each image sits in the view class its button reveals
        self.assertIn('class="ws ws0"', radios[0])
        self.assertRegex(h, r'<div class="wv wv0"><img src="data:image/png;base64,LONG"')
        self.assertRegex(h, r'<div class="wv wv1"><img src="data:image/png;base64,SHORT"')
        # and the CSS actually wires ws<k> -> wv<k> for every variant drawn
        for k in range(2):
            self.assertIn(f"input.ws{k}:checked ~ .wv{k}", sh.GLANCE_CSS)

    def test_two_figures_never_share_a_radio_group(self):
        a = sh._window_img([(200, "A"), (50, "B")], "x", "lnl")
        b = sh._window_img([(200, "A"), (50, "B")], "x", "leaves")
        self.assertFalse(set(re.findall(r'name="([^"]+)"', a)) & set(re.findall(r'name="([^"]+)"', b)))


if __name__ == "__main__":
    unittest.main()
