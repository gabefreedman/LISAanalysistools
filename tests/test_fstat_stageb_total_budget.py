"""FSTAT_STAGEB_TOTAL_MAX_GB: the whole-grid byte budget of stage B (24mo prep,
2026-11-05). Off by default (bit-identical); over budget, boxes are dropped
lowest F-per-GB first and the fit re-sized on the kept set.

Same harness as tests/test_fstat_mc_groups.py::StageBGroupingTest (the F-stat
scorer is mocked flat, so only the box bookkeeping and sizing are exercised).
"""
import os
import re
import unittest
from unittest import mock

import numpy as np

from lisatools.sampling.fstat_gridfit import run_stacked_stage_b

_KEYS = ("FSTAT_STAGEB_TOTAL_MAX_GB", "FSTAT_N_ALPHA", "FSTAT_N_SINDELTA",
         "FSTAT_PEAK_HALF_MHZ", "FSTAT_FDOT_AXIS", "FSTAT_N_MC", "FSTAT_N_F0",
         "FSTAT_N_PER_AXIS", "FSTAT_MC_GROUPING", "FSTAT_PEAKS_TO_FIT",
         "FSTAT_STAGEB_SKY_ADAPT", "FSTAT_STAGEB_GROUP_MAX_GB",
         "FSTAT_FDOT_RATIO_MAX")


def _stub_call_fstat(pr):
    n = len(pr)
    return np.ones(n), np.ones(n)


def _flat_fstat(N, M):
    return N * 0 + 1.0


class StageBTotalBudgetTest(unittest.TestCase):
    TOBS = 7776000.0
    MC_LIMS = [0.05, 1.0]
    BAND_EDGES = np.array([1.5e-3, 2.5e-3, 2.05e-2])
    # (f0 mHz, F, node, band): a cheap low-f box, a loud and a faint high-f box
    PEAKS = np.array([[2.0, 10.0, 0.0, 0], [20.0, 50.0, 0.0, 1], [20.2, 30.0, 0.0, 1]])

    def setUp(self):
        self._env = {k: os.environ.get(k) for k in _KEYS}
        for k in _KEYS:
            os.environ.pop(k, None)
        os.environ.update(FSTAT_N_ALPHA="2", FSTAT_N_SINDELTA="2",
                          FSTAT_PEAK_HALF_MHZ="0.0005",
                          # the production basis (launcher: FSTAT_FDOT_AXIS=1, ratio 5.0)
                          FSTAT_FDOT_AXIS="1", FSTAT_FDOT_RATIO_MAX="5.0")

    def tearDown(self):
        for k, v in self._env.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def _run(self, budget=None):
        if budget is not None:
            os.environ["FSTAT_STAGEB_TOTAL_MAX_GB"] = repr(float(budget))
        with mock.patch("lisatools.sampling.fstat_proposal.compute_fstat",
                        side_effect=_flat_fstat), \
                self.assertLogs("lisatools.sampling.fstat_gridfit", "INFO") as cm:
            out = run_stacked_stage_b(
                _stub_call_fstat, self.PEAKS.copy(), xp=np, Tobs=self.TOBS,
                band_edges_hz=self.BAND_EDGES, mc_lims=self.MC_LIMS)
        return out, "\n".join(cm.output)

    @staticmethod
    def _f0s(out):
        return sorted(np.round(np.asarray(out.f0_los) + 0.0005, 4).tolist())

    def _total_gb(self, log):
        return float(re.findall(r"\[stageB\] grid ([0-9.e+-]+) GB within", log)[-1])

    def test_off_by_default_keeps_every_box(self):
        out, log = self._run()
        self.assertEqual(len(out.f0_los), 3)
        self.assertNotIn("FSTAT_STAGEB_TOTAL_MAX_GB", log)

    def test_a_budget_above_the_grid_changes_nothing(self):
        ref, _ = self._run()
        out, log = self._run(budget=1e3)
        self.assertIn("within FSTAT_STAGEB_TOTAL_MAX_GB", log)
        self.assertEqual(self._f0s(out), self._f0s(ref))
        pts = np.array([[2.0, 0.5, 1.0, 0.0], [20.0, 0.5, 1.0, 0.0]])
        np.testing.assert_array_equal(np.asarray(out.logpdf(pts)), np.asarray(ref.logpdf(pts)))

    def test_over_budget_drops_the_faint_expensive_box_first(self):
        _, log = self._run(budget=1e3)
        total = self._total_gb(log)
        out, log = self._run(budget=0.99 * total)
        self.assertIn("keeping 2 of 3 peak boxes", log)
        # the cheap low-f box and the LOUD high-f box survive; F=30 at 20.2 goes
        self.assertEqual(self._f0s(out), [2.0, 20.0])
        self.assertLessEqual(self._total_gb(log), 0.99 * total)

    def test_a_tiny_budget_still_terminates_with_one_box(self):
        out, log = self._run(budget=1e-15)
        # one box = one group = the plain stack (box starts in ``_f0_lo``)
        self.assertEqual(len(out._f0_lo), 1)
        self.assertIn("with a single peak box left; fitting it anyway", log)


if __name__ == "__main__":
    unittest.main()
