"""scripts/fstat_proposal/extend_gb_band_floor.py: downward GB band-grid extension.

A tiny store in the production layout (``global_fit/sub_backend/gb`` with a
leading step axis on every per-band record) is built from a real
``GBState`` band record, extended by two bands, and read back through the
resume-side validation (``GBState.initialize_band_information``).
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts", "fstat_proposal"))

import extend_gb_band_floor as ext  # noqa: E402
from lisatools.globalfit.state import GBState, make_cap_edges  # noqa: E402

NW, NT, NSTEP, NLEAVES = 2, 3, 4, 6
DF = 1e-5
OLD = np.arange(10, 16) * DF            # 5 bands
NEW = np.arange(8, 16) * DF             # 7 bands: 2 prepended
K = 2


def _write_store(path, edges):
    cap = make_cap_edges(edges, 1, stagger=True)
    st = GBState(None)
    nb = len(edges) - 1
    st.initialize_band_information(
        NW, NT, edges, np.zeros((nb, NT)), cap_edges=cap, branch_name="gb",
        leaf_caps=True, search_shutoff_per_walker=True)
    bi = st.band_info
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        g = f.create_group("global_fit")
        g.attrs["iteration"] = NSTEP
        gb = g.create_group("sub_backend").create_group("gb")
        gb.attrs.update(dict(nwalkers=NW, ntemps=NT, num_bands=nb, num_cap_cells=nb, ndim=9))
        gb.create_dataset("band_edges", data=edges)
        gb.create_dataset("cap_edges", data=cap)
        for name in ext.BAND_AXIS:
            if name not in bi:
                continue
            arr = np.asarray(bi[name])
            # distinct values per band so a shift is visible
            data = np.stack([arr + 0 for _ in range(NSTEP)])
            if data.dtype.kind in "fi":
                data = data + rng.integers(1, 50, size=data.shape).astype(data.dtype)
            gb.create_dataset(name, data=data, maxshape=(None,) + data.shape[1:],
                              chunks=True, compression="gzip")
        for name in ext.STEP_SCALARS:
            gb.create_dataset(name, data=np.zeros((NSTEP, 1), dtype=np.int64))
        chain = np.zeros((NSTEP, NT, NW, NLEAVES, 9))
        chain[..., 1] = 0.5 * (edges[1] + edges[-2]) * 1e3     # f0 in mHz, inside the prior
        gb.create_dataset("chain", data=chain)
        gb.create_dataset("inds", data=np.ones((NSTEP, NT, NW, NLEAVES), dtype=bool))


def _run(*argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            rc = ext.main(list(argv))
        except SystemExit as e:
            rc = e.code
    return rc, buf.getvalue()


class ExtendGBBandFloorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = os.path.join(self.tmp.name, "gf_testing.h5")
        _write_store(self.store, OLD)
        self.edges = os.path.join(self.tmp.name, "new.npy")
        np.save(self.edges, NEW)
        with h5py.File(self.store, "r") as f:
            self.before = {k: f["global_fit/sub_backend/gb"][k][:]
                           for k in f["global_fit/sub_backend/gb"]}

    def tearDown(self):
        self.tmp.cleanup()

    def test_dry_run_writes_nothing(self):
        rc, out = _run(self.store, "--edges-npy", self.edges)
        self.assertEqual(rc, 0, out)
        self.assertIn("DRY RUN", out)
        with h5py.File(self.store, "r") as f:
            self.assertEqual(len(f["global_fit/sub_backend/gb/band_edges"]), len(OLD))

    def test_extension_shifts_old_bands_and_fills_new_ones_fresh(self):
        rc, out = _run(self.store, "--edges-npy", self.edges, "--apply", "--skip-resume-check")
        self.assertEqual(rc, 0, out)
        fills = ext.fresh_fill_values(NW, NT, 1, True)
        with h5py.File(self.store, "r") as f:
            gb = f["global_fit/sub_backend/gb"]
            np.testing.assert_array_equal(gb["band_edges"][:], NEW)
            np.testing.assert_array_equal(gb["cap_edges"][:], make_cap_edges(NEW, 1, True))
            self.assertEqual(int(gb.attrs["num_bands"]), len(NEW) - 1)
            self.assertIn(ext.STAMP, gb.attrs)
            for name, (axis, _grid) in ext.BAND_AXIS.items():
                if name not in self.before:
                    continue
                new = gb[name][:]
                self.assertEqual(new.shape[axis], self.before[name].shape[axis] + K, name)
                np.testing.assert_array_equal(
                    np.take(new, np.arange(K, new.shape[axis]), axis=axis), self.before[name],
                    err_msg=name)
                head = np.take(new, np.arange(K), axis=axis)
                if name == "band_temps":
                    want = np.repeat(np.take(self.before[name], [0], axis=axis), K, axis=axis)
                    np.testing.assert_array_equal(head, want)
                else:
                    self.assertTrue(np.all(head == fills[name]), name)
                self.assertIsNone(gb[name].maxshape[0], name)       # still growable
            np.testing.assert_array_equal(gb["chain"][:], self.before["chain"])

        # the resume-side validation accepts it on the new grid
        with h5py.File(self.store, "r") as f:
            gb = f["global_fit/sub_backend/gb"]
            bi = {k: gb[k][:] for k in gb if k not in ("chain", "inds")}
        bi["initialized"] = True
        st = GBState(None, band_info=bi)
        nt = st.initialize_band_information(
            NW, NT, NEW, np.zeros((len(NEW) - 1, NT)), cap_edges=make_cap_edges(NEW, 1, True),
            branch_name="gb", leaf_caps=True, search_shutoff_per_walker=True)
        self.assertEqual(nt, NT)
        self.assertEqual(st.band_info["num_bands"], len(NEW) - 1)

        # a second application is refused
        rc, out = _run(self.store, "--edges-npy", self.edges, "--apply", "--skip-resume-check")
        self.assertNotEqual(rc, 0)

    def test_refuses_a_grid_that_is_not_a_downward_extension(self):
        np.save(self.edges, np.arange(8, 16) * DF * 1.001)
        rc, out = _run(self.store, "--edges-npy", self.edges, "--apply", "--skip-resume-check")
        self.assertIn("not the top of the new one", str(rc) + out)

    def test_refuses_an_unknown_band_sized_dataset(self):
        with h5py.File(self.store, "r+") as f:
            f["global_fit/sub_backend/gb"].create_dataset(
                "band_new_thing", data=np.zeros((NSTEP, len(OLD) - 1)))
        rc, out = _run(self.store, "--edges-npy", self.edges, "--apply", "--skip-resume-check")
        self.assertIn("band_new_thing", str(rc))

    def test_refuses_to_apply_next_to_a_midit_checkpoint(self):
        open(self.store[:-3] + "_midit_checkpoint.pkl", "w").close()
        rc, out = _run(self.store, "--edges-npy", self.edges, "--apply", "--skip-resume-check")
        self.assertIn("blockers", str(rc))
        with h5py.File(self.store, "r") as f:
            self.assertEqual(len(f["global_fit/sub_backend/gb/band_edges"]), len(OLD))


if __name__ == "__main__":
    unittest.main()
