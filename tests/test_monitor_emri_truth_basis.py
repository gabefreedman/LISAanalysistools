"""The monitor's EMRI truth row must be in the run's EMRI sampling basis.

The stored EMRI chain is ``logm1, m2, a, p0, e0, dist, cosqS, phiS, cosqK,
phiK, Phi_phi0, Phi_r0`` with an ECLIPTIC sky (the engine converts the
catalogue's ICRS RA/Dec) and the raw catalogue spin angles. The monitor's
truth line must be built the same way, or the corner plots show bogus
offsets in the sky / spin columns.
"""

import unittest

import numpy as np

from lisatools.globalfit.monitor._short import _generator_functions
from lisatools.sources.emri.waveform import emri_catalogue_to_waveform_basis

ROW = {
    "PrimaryMassSSBFrame": 3.5e5,
    "SecondaryMassSSBFrame": 80.1,
    "PrimarySpinParameter": 0.95,
    "SemiLatusRectum": 7.389,
    "Eccentricity": 0.316,
    "InclinationAngle": 0.0,
    "LuminosityDistance": 4860.0,
    "RightAscension": 6.023,
    "Declination": -0.625,
    "PolarAnglePrimarySpin": 1.663,
    "AzimuthalAnglePrimarySpin": 1.743,
    "AzimuthalPhase": 2.4,
    "PolarPhase": 0.3,
    "RadialPhase": 0.974,
}


def _monitor_ns():
    ns = {"np": np}
    _generator_functions(ns, names=("_src_cat_to_sampling_emri",),
                         consts=("_SRC_BRANCH_SPECS",))
    return ns


class EmriTruthBasisTest(unittest.TestCase):
    def test_truth_matches_engine_sampling_basis(self):
        ns = _monitor_ns()
        got = ns["_src_cat_to_sampling_emri"](ROW)
        # engine physical row [m1, m2, a, p0, e0, xI0, dist, qS, phiS, qK, phiK,
        # Phi_phi0, Phi_theta0, Phi_r0] -> sampling basis via the stock EMRI
        # container's inverse ops (log m1, cos qS, cos qK; xI0 / Phi_theta0 filled)
        phys = np.asarray(emri_catalogue_to_waveform_basis(ROW), dtype=float)
        want = np.array([np.log(phys[0]), phys[1], phys[2], phys[3], phys[4], phys[6],
                         np.cos(phys[7]), phys[8], np.cos(phys[9]), phys[10],
                         phys[11], phys[13]])
        self.assertEqual(got.shape, (12,))
        # angles that the monitor wraps to [0, 2 pi)
        for k in (7, 9, 10, 11):
            want[k] %= 2 * np.pi
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)

    def test_param_labels_name_the_cosine_columns(self):
        ns = _monitor_ns()
        names = [n for n, _ in ns["_SRC_BRANCH_SPECS"]["emri"]["param_defs"]]
        self.assertEqual(names[6], "cosqS")
        self.assertEqual(names[8], "cosqK")


if __name__ == "__main__":
    unittest.main()
