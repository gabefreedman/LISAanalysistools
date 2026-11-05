"""VGB edge-band guard (2026-11-04).

The VGB move never proposes in the first or last band. On the 12mo run the
lowest VGB band [0.2778, 0.4167) mHz held 8 catalogue sources, which were
therefore never sampled. Covered here:

* ``vgb_guard_edge_bands``: splits an empty guard off an occupied edge band,
  leaves a clean grid bit-identical, refuses when no fine edge can split;
* ``VGBSetup.init_band_structure`` (uniform grid, ``VGB_BAND_LAYERS=8``) on
  the production WDM layer grid: every VGB ends in an interior band, and a
  catalogue with no low-f sources keeps the pre-fix grid exactly;
* ``extend_gb_band_floor.py --branch vgb``: the store split moves every
  per-band record up one band, gives the guard the fresh value, and the
  resume-side VGB band check accepts the config-built grid.
"""

import contextlib
import importlib
import io
import logging
import os
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "fstat_proposal"))

import extend_gb_band_floor as ext  # noqa: E402
from lisatools.domains import WDMSettings  # noqa: E402
from lisatools.globalfit.state import GBState  # noqa: E402

vgbmod = importlib.import_module("lisatools.globalfit.stock.erebor.vgb")

YR = 31558149.763545603
LDF = 1.0 / 7200.0                     # the production WDM layer_df (Nf * dt = 3600 s)
LOW_VGB_MHZ = [0.3077, 0.3408, 0.3418, 0.3420, 0.3812, 0.3815, 0.3830, 0.4089]


def _fine(k_lo, k_hi, div=8):
    return np.asarray([k * LDF / div for k in range(k_lo * div, k_hi * div + 1)])


def _setup(f0_mhz, min_freq=2.5e-4):
    """A VGBSetup with only the fields init_band_structure reads (no data load)."""
    dt, nf = 5.0, 720
    wdm = WDMSettings(nf, int(YR // (nf * dt)), dt, min_freq=min_freq, max_freq=2.5e-2)
    o = object.__new__(vgbmod.VGBSetup)
    f0 = np.asarray(f0_mhz, dtype=float)
    guard = 2.0 * 8 * float(wdm.layer_df)                       # prepare_vgb_branch's guard
    fields = dict(
        oversample=None, Tobs=YR, domain_settings=wdm, band_edges_override=None,
        start_freq=max(f0.min() * 1e-3 - guard, min_freq),
        end_freq=min(f0.max() * 1e-3 + guard, 2.5e-2),
        subband_divisor=8, band_edges_mode="uniform", band_unit_stride=2, band_layers=8,
        fdot_lims=[-1e-16, 1e-16], logger=logging.getLogger("test_vgb_guard"),
        chirp_mass_basis=True, sample_distance=True,
    )
    for k, v in fields.items():
        object.__setattr__(o, k, v)
    basis = vgbmod.vgb_fixed_basis(o)
    fp = np.zeros((f0.size, len(basis)))
    fp[:, basis.index("f0")] = f0
    object.__setattr__(o, "fixed_params", fp)
    return o


class GuardHelperTest(unittest.TestCase):
    def setUp(self):
        self.fine = _fine(2, 12)
        self.coarse = self.fine[::8]

    def test_low_edge_band_is_split_at_the_first_fine_edge(self):
        f0 = np.array(LOW_VGB_MHZ + [1.0]) * 1e-3
        out = vgbmod.vgb_guard_edge_bands(self.coarse, self.fine, f0)
        self.assertEqual(len(out), len(self.coarse) + 1)
        self.assertEqual(out[1], self.fine[1])
        np.testing.assert_array_equal(np.delete(out, 1), self.coarse)
        b = np.searchsorted(out, f0, side="right") - 1
        self.assertTrue(np.all(b >= 1) and np.all(b <= len(out) - 3))

    def test_clean_grid_is_returned_unchanged(self):
        f0 = np.array([0.6, 1.0, 1.2]) * 1e-3
        out = vgbmod.vgb_guard_edge_bands(self.coarse, self.fine, f0)
        np.testing.assert_array_equal(out, self.coarse)

    def test_high_edge_band_is_split_too(self):
        f0 = np.array([0.6e-3, self.coarse[-2] + 0.2 * LDF])
        out = vgbmod.vgb_guard_edge_bands(self.coarse, self.fine, f0)
        self.assertEqual(out[-2], self.fine[-2])
        b = np.searchsorted(out, f0, side="right") - 1
        self.assertLessEqual(b.max(), len(out) - 3)

    def test_refuses_when_no_fine_edge_can_split(self):
        # one fine band per coarse band (VGB_BAND_LAYERS=1): nothing to split
        with self.assertRaises(ValueError):
            vgbmod.vgb_guard_edge_bands(self.fine, self.fine, np.array([self.fine[0] + 1e-7]))
        # a source inside the would-be guard band
        with self.assertRaises(ValueError):
            vgbmod.vgb_guard_edge_bands(self.coarse, self.fine,
                                        np.array([0.5 * (self.fine[0] + self.fine[1])]))


class VGBSetupGridTest(unittest.TestCase):
    def test_low_vgbs_land_in_an_interior_band(self):
        f0 = LOW_VGB_MHZ + [1.0, 2.0, 6.1]
        o = _setup(f0)
        o.init_band_structure()
        e = np.asarray(o.band_edges)
        np.testing.assert_allclose(e[:4] * 1e3, [0.277778, 0.295139, 0.416667, 0.555556],
                                   rtol=0, atol=1e-6)
        self.assertEqual(e[1], 17 * LDF / 8)       # the GB grid's second edge, bit for bit
        b = np.searchsorted(e, np.asarray(f0) * 1e-3, side="right") - 1
        self.assertTrue(np.all(b[:8] == 1), b)
        self.assertEqual(o.num_sub_bands, len(e) - 1)
        self.assertEqual(len(o.band_N_vals), len(e) - 1)
        self.assertEqual(o.f0_lims[0], e[1])

    def test_catalogue_clear_of_the_edges_keeps_the_old_grid(self):
        f0 = [0.9, 2.0, 6.1]
        o = _setup(f0)
        o.init_band_structure()
        e = np.asarray(o.band_edges)
        # the pre-fix grid: every 8th fine edge
        fine = _fine(int(np.ceil(o.start_freq / LDF)), int(np.floor(o.end_freq / LDF)))
        keep = fine[::8]
        if keep[-1] != fine[-1]:
            keep = np.append(keep, fine[-1])
        np.testing.assert_array_equal(e, keep)


NW, NT, NSTEP = 2, 3, 4
NLEAVES = 5


def _write_vgb_store(path, edges, gb_edges, low_count=3):
    st = GBState(None)
    nb = len(edges) - 1
    st.initialize_band_information(NW, NT, edges, np.zeros((nb, NT)), branch_name="vgb",
                                   leaf_caps=False)
    bi = st.band_info
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = NSTEP
        sb = g.create_group("sub_backend")
        gb = sb.create_group("gb")
        gb.create_dataset("band_edges", data=gb_edges)
        v = sb.create_group("vgb")
        v.attrs.update(dict(nwalkers=NW, ntemps=NT, num_bands=nb, num_cap_cells=nb, ndim=6))
        v.create_dataset("band_edges", data=edges)
        v.create_dataset("cap_edges", data=edges)
        for name in ext.BAND_AXIS:
            if name not in bi:
                continue
            arr = np.asarray(bi[name])
            data = np.stack([arr + 0 for _ in range(NSTEP)])
            if data.dtype.kind in "fi" and name != "band_num_binaries":
                data = data + rng.integers(1, 50, size=data.shape).astype(data.dtype)
            v.create_dataset(name, data=data, maxshape=(None,) + data.shape[1:], chunks=True)
        nbin = np.zeros((NSTEP, NT, NW, nb))
        nbin[..., 0] = low_count                       # the stranded sources
        nbin[..., 2] = NLEAVES - low_count
        v["band_num_binaries"][:] = nbin
        for name in ("band_shutoff_epoch", "band_shutoff_since_revive"):
            v.create_dataset(name, data=np.zeros((NSTEP, 1), dtype=np.int64))
        v.create_dataset("chain", data=np.ones((NSTEP, NT, NW, NLEAVES, 6)))
        v.create_dataset("inds", data=np.ones((NSTEP, NT, NW, NLEAVES), dtype=bool))


def _run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = ext.main(list(argv))
        except SystemExit as e:
            rc = e.code
    return rc, buf.getvalue()


class VGBStoreSplitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = os.path.join(self.tmp.name, "gf_testing.h5")
        fine = _fine(2, 8)
        self.old = fine[::8]                            # 6 one-layer bands from 0.2778 mHz
        self.gb = fine                                  # GB grid on the same fine edges
        _write_vgb_store(self.store, self.old, self.gb)
        with h5py.File(self.store, "r") as f:
            self.before = {k: f["global_fit/sub_backend/vgb"][k][:]
                           for k in f["global_fit/sub_backend/vgb"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_writes_nothing(self):
        rc, out = _run(self.store, "--branch", "vgb", "--guard-from-gb-grid")
        self.assertEqual(rc, 0, out)
        self.assertIn("DRY RUN", out)
        self.assertIn("[3, 3]", out)                    # the stranded sources are reported
        with h5py.File(self.store, "r") as f:
            self.assertEqual(len(f["global_fit/sub_backend/vgb/band_edges"]), len(self.old))

    def test_split_moves_records_up_one_and_resume_check_accepts_it(self):
        rc, out = _run(self.store, "--branch", "vgb", "--guard-from-gb-grid", "--apply",
                       "--skip-resume-check")
        self.assertEqual(rc, 0, out)
        new = np.insert(self.old, 1, self.gb[1])
        fills = ext.fresh_fill_values_vgb(NW, NT)
        with h5py.File(self.store, "r") as f:
            v = f["global_fit/sub_backend/vgb"]
            np.testing.assert_array_equal(v["band_edges"][:], new)
            np.testing.assert_array_equal(v["cap_edges"][:], new)
            self.assertEqual(int(v.attrs["num_bands"]), len(new) - 1)
            self.assertIn(ext.VGB_STAMP, v.attrs)
            for name, (axis, _grid) in ext.BAND_AXIS.items():
                if name not in self.before:
                    continue
                arr = v[name][:]
                np.testing.assert_array_equal(
                    np.take(arr, np.arange(1, arr.shape[axis]), axis=axis), self.before[name],
                    err_msg=name)
                head = np.take(arr, [0], axis=axis)
                if name == "band_temps":
                    np.testing.assert_array_equal(head, np.take(self.before[name], [0], axis=axis))
                else:
                    self.assertTrue(np.all(head == fills[name]), name)
            nbin = v["band_num_binaries"][-1]
            self.assertFalse(nbin[..., 0].any())        # the guard is empty
            self.assertTrue(np.all(nbin[..., 1] == 3))  # the sources are in band 1 now
            np.testing.assert_array_equal(v["chain"][:], self.before["chain"])

            bi = {k: v[k][:] for k in v if k not in ("chain", "inds")}
        bi["initialized"] = True
        st = GBState(None, band_info=bi)
        nt = st.initialize_band_information(NW, NT, new, np.zeros((len(new) - 1, NT)),
                                            branch_name="vgb", leaf_caps=False)
        self.assertEqual(nt, NT)
        self.assertEqual(st.band_info["num_bands"], len(new) - 1)

        # the un-split grid is refused by the same check, and a second split is refused
        bi2 = {k: self.before[k] for k in self.before if k not in ("chain", "inds")}
        bi2["initialized"] = True
        with self.assertRaises(ValueError):
            GBState(None, band_info=bi2).initialize_band_information(
                NW, NT, new, np.zeros((len(new) - 1, NT)), branch_name="vgb", leaf_caps=False)
        rc, out = _run(self.store, "--branch", "vgb", "--guard-from-gb-grid", "--apply",
                       "--skip-resume-check")
        self.assertNotEqual(rc, 0)

    def test_refuses_when_gb_grid_does_not_share_the_floor(self):
        with h5py.File(self.store, "r+") as f:
            del f["global_fit/sub_backend/gb/band_edges"]
            f["global_fit/sub_backend/gb"].create_dataset("band_edges", data=self.gb[8:])
        rc, out = _run(self.store, "--branch", "vgb", "--guard-from-gb-grid", "--apply",
                       "--skip-resume-check")
        self.assertIn("not the VGB fine edge", str(rc))

    def test_refuses_an_edge_that_is_not_inside_the_lowest_band(self):
        bad = os.path.join(self.tmp.name, "bad.npy")
        np.save(bad, np.insert(self.old, 2, 0.5 * (self.old[1] + self.old[2])))
        rc, out = _run(self.store, "--branch", "vgb", "--edges-npy", bad, "--apply",
                       "--skip-resume-check")
        self.assertIn("inside its lowest band", str(rc))


if __name__ == "__main__":
    unittest.main()
