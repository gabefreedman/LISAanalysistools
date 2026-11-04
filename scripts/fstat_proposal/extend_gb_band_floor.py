"""Extend a global-fit store's GB band grid DOWNWARD so the run resumes where it left off.

Use case (2026-11-03): the first 12mo run (``gf_prod_12mo_flagship``, in
``full_pe``) is continued with a lower GB frequency floor (``GB_MIN_FREQ``
5.5e-4 -> 2.5e-4). On the uniform WDM band grid (``GB_BAND_EDGES_MODE``
uniform, ``GB_SUBBAND_DIVISOR=8``) lowering the floor only PREPENDS sub-bands
(1232 -> 1248, 16 new bands at the bottom); every existing edge keeps its exact
value. So every per-band array can be carried over EXACTLY, shifted by the
number of new bands, instead of being reset the way
``migrate_gb_band_edges.py`` (written for re-gridding) does.

What this script does to ``<group>/sub_backend/gb``, in place, ONLY with
``--apply`` (the default is a dry run that prints the plan):

* ``band_edges`` / ``cap_edges`` -> the new grids (validated: the old grid
  must be the TOP of the new one, value for value);
* every per-band and per-cap-cell dataset (every stored row) -> the old
  values moved up by ``k`` bands (``k`` cells), the ``k`` new leading entries
  set to the FRESH-START value the run itself would allocate. The fresh
  values are NOT hard-coded here: they are read off
  ``GBState.initialize_band_information`` with every record family enabled,
  so they cannot drift from the code that will read them;
* ``band_temps`` -> the new bands copy the lowest old band's ladder (the
  fresh-start value, zeros, is only a placeholder the move builder
  overwrites on a FRESH start; a resumed ladder is never rebuilt);
* group attrs ``num_bands`` / ``num_cap_cells`` -> the new counts, plus a
  ``band_floor_extended`` stamp so a second application is refused.

Leaves (``chain`` / ``inds`` / ``d_h`` / ``h_h``) are untouched: a source is
assigned to a band by ``searchsorted(band_edges, f0)`` at propose time, and
every source of the old run sits above the old floor. Nothing outside
``sub_backend/gb`` is touched (the VGB grid is catalogue-built and does not
depend on ``GB_MIN_FREQ``; ``--edges-from`` verifies that).

REFUSES (never guesses): a grid that is not a pure downward extension, a
band-sized dataset it has no rule for, an already-extended store, a VGB grid
that differs from the reference store's, alive sources outside the new f0
prior, and (with ``--apply``) a run folder that still holds a mid-iteration
checkpoint or a running backup copy of the UN-extended store (``run.py``
would load either before the band check). The stale F-stat epoch
directories are reported (the move refits them itself; moving them aside
is cleaner).

Writes are staged: every new dataset is written under a temporary name
first, and only when all of them are complete are the old ones swapped
out. RUN IT ON A COPY OF THE STORE, never on the only copy.

Usage::

    # new grids from a store built with the target config (preferred):
    python scripts/fstat_proposal/extend_gb_band_floor.py <copy>/gf_prod_12mo_testing.h5 \\
        --edges-from /shared/data/global_fit_output/gf_prod_12mo_v1.2.8dev/gf_prod_12mo_testing.h5
    # ... prints the plan; then the same command with --apply.
"""

import argparse
import glob
import json
import os
import sys

import h5py
import numpy as np

#: datasets in sub_backend/gb that carry no band axis (left untouched). The
#: (step, 1) scalars are checked to really be scalars before being trusted.
NOT_BANDED = ("chain", "inds", "d_h", "h_h", "source_uid")
STEP_SCALARS = ("band_shutoff_epoch", "band_shutoff_since_revive",
                "band_shutoff_w_step", "band_window_rule")
STATIC_GRIDS = ("band_edges", "cap_edges")

#: band-axis position of every per-band / per-cell dataset as STORED (the
#: leading step axis included) and the grid it is indexed on. Mirrors
#: GBState.initialize_band_information + the ensure_* helpers in
#: lisatools/globalfit/state.py; a dataset not listed here whose shape holds
#: the old band or cell count is refused.
BAND_AXIS = {
    # name: (axis, grid)
    "band_temps": (1, "band"),
    "band_swaps_proposed": (1, "band"),
    "band_swaps_accepted": (1, "band"),
    "band_num_proposed": (1, "band"),
    "band_num_accepted": (1, "band"),
    "band_num_proposed_rj": (1, "band"),
    "band_num_accepted_rj": (1, "band"),
    "band_num_binaries": (3, "band"),
    "band_leaf_cap": (1, "band"),
    "band_cap_iters": (1, "band"),
    "band_best_ll": (1, "band"),
    "band_cold_ll": (2, "band"),
    "band_best_ll_w": (2, "band"),
    "band_occ_streak": (1, "band"),
    "band_occ_last": (1, "band"),
    "band_rj_shutoff": (1, "band"),
    "band_stage": (1, "band"),
    "band_stage_w": (2, "band"),
    "band_stage_occ_last_w": (2, "band"),
    "band_stage_streak_w": (2, "band"),
    "band_rj_shutoff_w": (2, "band"),
    "band_shutoff_best_w": (2, "band"),
    "band_shutoff_streak_w": (2, "band"),
    "band_cold_logl_max_w": (2, "band"),
    "band_cold_logl_w": (2, "band"),
    "band_shutoff_reset_w": (2, "band"),
    "band_cold_logl_peak_w": (2, "band"),
    "cap_cell_leaf_cap": (1, "cap"),
    "cap_cell_iters": (1, "cap"),
    "cap_cell_best_ll": (1, "cap"),
    "cap_cell_cold_ll": (2, "cap"),
    "cap_cell_leaf_cap_w": (2, "cap"),
    "cap_cell_iters_w": (2, "cap"),
    "cap_cell_best_ll_w": (2, "cap"),
}

#: GB sampling basis column of f0, in mHz (recipe.py: the warm floor box is
#: ``band_edges[...] * 1e3`` at column 1).
F0_COL, F0_SCALE = 1, 1e3

STAMP = "band_floor_extended"
TMP_SUFFIX = "__floorext_tmp"
ROW_BLOCK = 64


def fresh_fill_values(nwalkers, ntemps, cap_divisor, cap_stagger):
    """The per-band / per-cell FRESH-START value of every record, from the code.

    Builds a fresh 3-band ``GBState`` band record with every optional record
    family switched on and reads the value at band (cell) 0. Every fresh
    array is constant along its band axis, which is checked.

    Returns:
        dict: ``{dataset name: scalar}``.
    """
    from lisatools.globalfit.state import GBState, make_cap_edges

    edges = np.array([1.0, 2.0, 3.0, 4.0])
    cap = make_cap_edges(edges, cap_divisor, stagger=cap_stagger)
    st = GBState(None)
    st.initialize_band_information(
        nwalkers, ntemps, edges, np.zeros((3, ntemps)), cap_edges=cap,
        branch_name="gb", leaf_caps=True, leaf_cap_per_walker=True,
        search_stage_per_walker=True, search_shutoff_per_walker=True,
    )
    bi = st.band_info
    out = {}
    for name, (axis, _grid) in BAND_AXIS.items():
        if name not in bi:
            raise SystemExit(
                f"the fresh GB band record has no {name!r}: the state layout "
                "changed, update BAND_AXIS in this script before using it.")
        arr = np.asarray(bi[name])
        bare_axis = axis - 1                     # no step axis in band_info
        first = np.take(arr, [0], axis=bare_axis)
        if not np.array_equal(np.broadcast_to(first, arr.shape), arr, equal_nan=True):
            raise SystemExit(f"fresh {name!r} is not constant along its band axis")
        vals = np.unique(first)
        if vals.size != 1:
            raise SystemExit(f"fresh {name!r} has several values {vals}; no scalar fill")
        out[name] = vals[0].item()
    return out


def read_grids(path):
    with h5py.File(path, "r") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        sb = f[group]["sub_backend"]
        gb = sb["gb"]
        be = np.asarray(gb["band_edges"][:], dtype=float)
        ce = np.asarray(gb["cap_edges"][:], dtype=float) if "cap_edges" in gb else None
        vgb = (np.asarray(sb["vgb"]["band_edges"][:], dtype=float)
               if "vgb" in sb and "band_edges" in sb["vgb"] else None)
    return be, ce, vgb


def plan_datasets(gb, nb_old, nc_old):
    """Classify every dataset of the gb group; refuse anything unknown that is band-sized."""
    banded, untouched, refused = [], [], []
    for name in sorted(gb.keys()):
        obj = gb[name]
        if not isinstance(obj, h5py.Dataset):
            refused.append(f"{name}: a group, not a dataset")
            continue
        shp = tuple(obj.shape)
        if name.endswith(TMP_SUFFIX):
            refused.append(f"{name}: leftover temporary from an interrupted run of "
                           "this script (start again from a fresh copy)")
            continue
        if name in STATIC_GRIDS:
            continue
        if name in NOT_BANDED:
            untouched.append(name)
            continue
        if name in STEP_SCALARS:
            if len(shp) != 2 or shp[1] != 1:
                refused.append(f"{name}: expected (step, 1), got {shp}")
            else:
                untouched.append(name)
            continue
        if name in BAND_AXIS:
            axis, grid = BAND_AXIS[name]
            want = nb_old if grid == "band" else nc_old
            if len(shp) <= axis or shp[axis] != want:
                refused.append(f"{name}: shape {shp}, expected the {grid} axis at "
                               f"{axis} with length {want}")
            else:
                banded.append(name)
            continue
        if nb_old in shp[1:] or nc_old in shp[1:]:
            refused.append(f"{name}: shape {shp} holds the band/cell count but this "
                           "script has no rule for it (add it to BAND_AXIS)")
        else:
            untouched.append(name)
    return banded, untouched, refused


def extend_block(block, axis, k, fill, edge_copy):
    """Prepend ``k`` entries along ``axis`` of one row block."""
    if edge_copy:
        pad = np.repeat(np.take(block, [0], axis=axis), k, axis=axis)
    else:
        shp = list(block.shape)
        shp[axis] = k
        pad = np.full(shp, fill, dtype=block.dtype)
    return np.concatenate([pad, block], axis=axis)


def write_extended(gb, name, axis, k, fill, edge_copy):
    """Write ``name`` + ``TMP_SUFFIX`` = the extended copy of ``name``, row block by row block."""
    src = gb[name]
    shp = list(src.shape)
    shp[axis] += k
    chunks = None
    if src.chunks is not None:
        chunks = list(src.chunks)
        # a chunk spanning the whole band axis keeps spanning it
        if chunks[axis] >= src.shape[axis]:
            chunks[axis] = shp[axis]
        chunks = tuple(chunks)
    maxshape = None
    if src.maxshape is not None:
        maxshape = list(src.maxshape)
        if maxshape[axis] is not None:
            maxshape[axis] = shp[axis]
        maxshape = tuple(maxshape)
    tmp = name + TMP_SUFFIX
    dst = gb.create_dataset(
        tmp, shape=tuple(shp), dtype=src.dtype, chunks=chunks, maxshape=maxshape,
        compression=src.compression, compression_opts=src.compression_opts,
        shuffle=src.shuffle, fletcher32=src.fletcher32,
    )
    for k0 in range(0, src.shape[0], ROW_BLOCK):
        k1 = min(src.shape[0], k0 + ROW_BLOCK)
        dst[k0:k1] = extend_block(src[k0:k1], axis, k, fill, edge_copy)
    for a, v in src.attrs.items():
        dst.attrs[a] = v
    return tmp


def replace_static(gb, name, arr):
    src = gb[name]
    kw = dict(compression=src.compression, compression_opts=src.compression_opts)
    del gb[name]
    gb.create_dataset(name, data=np.asarray(arr, dtype=float), **kw)


def verify_resume(path, band_edges, cap_edges, search_shutoff_per_walker=True):
    """Read ``path`` the way a RESUME does and run the GB band checks on ``band_edges``.

    ``GFHDFBackend.get_last_sample`` (format gate included) with the
    production branch classes, then ``GBState.initialize_band_information``
    with the run's grids -- the call the GB move builder makes on a resume,
    whose band-grid / cap-grid / shape checks are the ones that refuse an
    un-migrated store. Any warning that a record was reset or discarded is a
    failure: an extended store must restore every record as it is.

    Returns:
        int: 0 on success (raises / exits otherwise).
    """
    import logging

    from lisatools.globalfit.hdfbackend import (
        EMRIHDFBackend, GBHDFBackend, GFHDFBackend, MBHHDFBackend, ModuleSubBackend,
        SOBBHHDFBackend)
    from lisatools.globalfit.state import (
        EMRIState, GBState, MBHState, ModuleSubState, SOBBHState)

    special = {"gb": (GBState, GBHDFBackend), "vgb": (GBState, GBHDFBackend),
               "mbh": (MBHState, MBHHDFBackend), "emri": (EMRIState, EMRIHDFBackend),
               "sobbh": (SOBBHState, SOBBHHDFBackend)}
    with h5py.File(path, "r") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f[group].attrs["branch_names"]]
        nwalkers = int(f[group]["sub_backend"]["gb"].attrs["nwalkers"])
        ntemps = int(f[group]["sub_backend"]["gb"].attrs["ntemps"])
    states = {n: special.get(n, (ModuleSubState, ModuleSubBackend))[0] for n in names}
    backends = {n: special.get(n, (ModuleSubState, ModuleSubBackend))[1] for n in names}

    seen = []

    class _Grab(logging.Handler):
        def emit(self, rec):
            if rec.levelno >= logging.WARNING:
                seen.append(rec.getMessage())

    handler = _Grab()
    lg = logging.getLogger("lisatools")
    lg.addHandler(handler)
    try:
        be = GFHDFBackend(path, sub_state_bases=states, sub_backend=backends)
        be.check_format_version("resume")
        state = be.get_last_sample()
        nt = state.sub_states["gb"].initialize_band_information(
            nwalkers, ntemps, band_edges, np.zeros((len(band_edges) - 1, ntemps)),
            cap_edges=cap_edges, branch_name="gb", leaf_caps=True,
            leaf_cap_per_walker=False, search_stage_per_walker=False,
            search_shutoff_per_walker=search_shutoff_per_walker)
    finally:
        lg.removeHandler(handler)
    bi = state.sub_states["gb"].band_info
    bad = [w for w in seen if any(s in w.lower() for s in (
        "reset", "discard", "restart", "reopen", "mismatch", "corrupt"))]
    print(f"resume check: stored iteration {be.iteration}; gb ntemps {nt}, "
          f"num_bands {bi['num_bands']}, num_cap_cells {bi['num_cap_cells']}; "
          f"{len(seen)} warning(s), {len(bad)} reset/discard-type")
    for w in bad:
        print("  !!", w[:300])
    if bad or nt != ntemps or bi["num_bands"] != len(band_edges) - 1:
        raise SystemExit("RESUME CHECK FAILED")
    print("resume check: OK (the run's resume read and GB band/cap checks pass)")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("h5_path", help="the COPY of the store to extend in place")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--edges-from", help="a store built with the TARGET config; its "
                     "gb band_edges / cap_edges are used and its vgb grid is compared")
    src.add_argument("--edges-npy", help=".npy of the new gb band edges (Hz), e.g. the "
                     "gb_config_band_edges.npy run.py writes beside a refused store; the "
                     "cap grid is then built with --cap-divisor / --cap-stagger")
    p.add_argument("--cap-divisor", type=int,
                   default=int(os.environ.get("GB_CAP_DIVISOR", "1")))
    p.add_argument("--cap-stagger", type=int, choices=(0, 1),
                   default=int(os.environ.get("GB_CAP_STAGGER", "1")))
    p.add_argument("--apply", action="store_true",
                   help="write the changes (default: dry run, nothing written)")
    p.add_argument("--verify-resume", action="store_true",
                   help="only run the resume check on an ALREADY extended store "
                        "(--apply runs it automatically at the end)")
    p.add_argument("--skip-resume-check", action="store_true",
                   help="unit tests only (a toy store is not a full global-fit file)")
    args = p.parse_args(argv)

    from lisatools.globalfit.state import make_cap_edges

    path = os.path.abspath(args.h5_path)
    if args.verify_resume:
        if args.edges_from:
            vb, vc, _ = read_grids(args.edges_from)
        else:
            vb = np.asarray(np.load(args.edges_npy), dtype=float)
            vc = make_cap_edges(vb, args.cap_divisor, stagger=bool(args.cap_stagger))
        return verify_resume(path, vb, vc)
    old_be, old_ce, old_vgb = read_grids(path)
    nb_old = len(old_be) - 1
    if old_ce is None:
        raise SystemExit("the store has no gb cap_edges; this script expects the "
                         "current (cap-grid) layout")
    nc_old = len(old_ce) - 1

    # ---- the new grids -------------------------------------------------------
    if args.edges_from:
        new_be, new_ce, ref_vgb = read_grids(args.edges_from)
        if new_ce is None:
            raise SystemExit(f"{args.edges_from} has no gb cap_edges")
        if old_vgb is not None or ref_vgb is not None:
            if (old_vgb is None or ref_vgb is None or old_vgb.shape != ref_vgb.shape
                    or not np.allclose(old_vgb, ref_vgb, rtol=1e-12, atol=0.0)):
                raise SystemExit("the VGB band grid differs between this store and "
                                 "the reference; the VGB branch would need its own "
                                 "migration. Refusing.")
        derived = make_cap_edges(new_be, args.cap_divisor, stagger=bool(args.cap_stagger))
        if derived.shape != new_ce.shape or not np.allclose(derived, new_ce, rtol=1e-12, atol=0):
            raise SystemExit("the reference store's cap_edges are not make_cap_edges("
                             f"band_edges, {args.cap_divisor}, stagger={bool(args.cap_stagger)}); "
                             "pass the --cap-divisor / --cap-stagger the run uses")
    else:
        new_be = np.asarray(np.load(args.edges_npy), dtype=float)
        new_ce = make_cap_edges(new_be, args.cap_divisor, stagger=bool(args.cap_stagger))
    nb_new, nc_new = len(new_be) - 1, len(new_ce) - 1
    k, kc = nb_new - nb_old, nc_new - nc_old

    print(f"store      : {path}")
    print(f"old gb grid: {nb_old} bands [{old_be[0]*1e3:.6f}, {old_be[-1]*1e3:.6f}] mHz, "
          f"{nc_old} cap cells")
    print(f"new gb grid: {nb_new} bands [{new_be[0]*1e3:.6f}, {new_be[-1]*1e3:.6f}] mHz, "
          f"{nc_new} cap cells")

    if not np.all(np.diff(new_be) > 0):
        raise SystemExit("the new band edges are not strictly ascending")
    if k <= 0:
        raise SystemExit(f"the new grid has {nb_new} bands, not more than {nb_old}: "
                         "this script only extends the grid DOWNWARD")
    if not np.allclose(new_be[k:], old_be, rtol=1e-12, atol=0.0):
        raise SystemExit("the old band grid is not the top of the new one (edges differ): "
                         "not a pure downward extension; use migrate_gb_band_edges.py")
    if kc <= 0 or not np.allclose(new_ce[kc + 1:], old_ce[1:], rtol=1e-12, atol=0.0):
        raise SystemExit("the old cap grid is not the top of the new one above its first "
                         "edge: not a pure downward extension of the cap cells")
    print(f"extension  : +{k} bands, +{kc} cap cells at the bottom; every old edge "
          "is unchanged")

    with h5py.File(path, "r") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        g = f[group]
        gb = g["sub_backend"]["gb"]
        if STAMP in gb.attrs:
            raise SystemExit(f"already extended: {gb.attrs[STAMP]}")
        nwalkers = int(gb.attrs["nwalkers"])
        ntemps = int(gb.attrs["ntemps"])
        it = int(g.attrs["iteration"])
        banded, untouched, refused = plan_datasets(gb, nb_old, nc_old)
        # the alive sources of the last stored row must sit inside the new f0 prior
        row = it - 1
        ch = gb["chain"][row]
        ind = gb["inds"][row].astype(bool)
        f0 = ch[..., F0_COL][ind] / F0_SCALE
    if refused:
        raise SystemExit("REFUSING, unknown or malformed datasets:\n  " + "\n  ".join(refused))
    lo, hi = new_be[1], new_be[-2]          # f0_lims = interior edges (gb.py)
    n_out = int(np.sum((f0 < lo) | (f0 > hi)))
    print(f"last row   : {it} stored rows; {f0.size} alive leaves over every rung; "
          f"f0 in [{f0.min()*1e3:.4f}, {f0.max()*1e3:.4f}] mHz; new f0 prior "
          f"[{lo*1e3:.4f}, {hi*1e3:.4f}] mHz; outside: {n_out}")
    if n_out:
        raise SystemExit("alive sources outside the new f0 prior -- refusing")

    fills = fresh_fill_values(nwalkers, ntemps, args.cap_divisor, bool(args.cap_stagger))
    print(f"datasets   : {len(banded)} per-band/per-cell to extend, "
          f"{len(untouched)} untouched ({', '.join(untouched)})")
    for name in banded:
        axis, grid = BAND_AXIS[name]
        how = ("copy of the lowest old band" if name == "band_temps"
               else f"fresh value {fills[name]!r}")
        print(f"  {name:26s} {grid:4s} axis {axis}: +{k if grid == 'band' else kc} "
              f"leading entries = {how}")

    # ---- the run folder ------------------------------------------------------
    run_dir = os.path.dirname(path)
    stem = path[:-3]
    blockers = [q for q in (stem + "_midit_checkpoint.pkl", stem + "_running_backup_copy.h5")
                if os.path.exists(q)]
    stale = sorted(glob.glob(os.path.join(run_dir, "gb_fstat_fit", "shared*", "epoch_*")))
    if stale:
        print(f"F-stat     : {len(stale)} epoch dir(s) under {run_dir}/gb_fstat_fit/ were "
              "fitted on the OLD grid; the move refits them itself, but move the "
              "shared* dirs aside so nothing stale is ever read")
    if blockers:
        print("BLOCKERS   : these hold the UN-extended state and run.py reads them "
              "BEFORE the band check -- move them aside:\n  " + "\n  ".join(blockers))

    if not args.apply:
        print("\nDRY RUN: nothing written. Re-run with --apply.")
        return 0
    if blockers:
        raise SystemExit("refusing to --apply while the blockers above exist")

    # ---- write ---------------------------------------------------------------
    with h5py.File(path, "r+") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        gb = f[group]["sub_backend"]["gb"]
        tmps = []
        for name in banded:
            axis, grid = BAND_AXIS[name]
            tmps.append((name, write_extended(
                gb, name, axis, k if grid == "band" else kc, fills.get(name),
                edge_copy=(name == "band_temps"))))
            print(f"  staged {name}")
        # every staged copy is complete: swap them in
        for name, tmp in tmps:
            del gb[name]
            gb.move(tmp, name)
        replace_static(gb, "band_edges", new_be)
        replace_static(gb, "cap_edges", new_ce)
        gb.attrs["num_bands"] = nb_new
        gb.attrs["num_cap_cells"] = nc_new
        gb.attrs[STAMP] = json.dumps({
            "bands": [nb_old, nb_new], "cap_cells": [nc_old, nc_new],
            "old_floor_hz": float(old_be[0]), "new_floor_hz": float(new_be[0]),
            "edges_from": args.edges_from or args.edges_npy,
        })

    # ---- verify ----------------------------------------------------------------
    with h5py.File(path, "r") as f:
        group = "global_fit" if "global_fit" in f else "mcmc"
        gb = f[group]["sub_backend"]["gb"]
        assert np.allclose(gb["band_edges"][:], new_be, rtol=0, atol=0)
        assert np.allclose(gb["cap_edges"][:], new_ce, rtol=0, atol=0)
        for name in banded:
            axis, grid = BAND_AXIS[name]
            want = nb_new if grid == "band" else nc_new
            assert gb[name].shape[axis] == want, (name, gb[name].shape)
    print(f"done: {nb_old} -> {nb_new} bands, {nc_old} -> {nc_new} cap cells; "
          f"stamp {STAMP!r} written. Resume with GB_MIN_FREQ matching these edges.")
    if args.skip_resume_check:
        return 0
    return verify_resume(path, new_be, new_ce)


if __name__ == "__main__":
    sys.exit(main())
