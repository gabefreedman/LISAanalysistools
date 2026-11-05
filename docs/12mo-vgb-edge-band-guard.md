# Runbook: let the 12mo run sample its 8 lowest-frequency VGBs

*2026-11-04. Resume `gf_prod_12mo_flagship` in `full_pe` with one empty guard
band split off the bottom of the VGB band grid, so the 8 verification binaries
at 0.31-0.41 mHz are sampled for the first time.*

## The bug

The VGB move never proposes in the first or last band (`gbspecialstretch.py`,
`band_inds > 0 & < num_bands - 1`; tempering uses `arange(1, nb - 1)` too).
`prepare_vgb_branch` asks for one empty guard band below the lowest VGB, but
clamps the guard start to `MIN_FREQ`. With `MIN_FREQ=2.5e-4` and the lowest VGB
at 0.31 mHz the guard collapsed and the VGB grid starts at the data floor:
band 0 = [0.2778, 0.4167) mHz holds VW Hyi, EX Hya, HD265435, SDSSJ1337,
WD0957, J0221, J2013 and WZ Sge.

Evidence (Nov 1 extract, rows 1117-1128): VGB band 0 got 0 in-model
proposals, every other band ~1,100 per leaf; the 8 leaves (39, 40, 49-54)
have `h_h` = NaN on every row; their phi0 / cos_iota / psi are bit-identical
from row 5 to row 1128 (only the ridge-Gibbs coordinates dist / Mc / ratio
move). Their templates started at the catalogue truth, so the residual is
probably fine, but **their posteriors so far are the injection, not a fit**.

## The fix

* `vgb.py::vgb_guard_edge_bands` (called from `VGBSetup.init_band_structure`,
  uniform grid): an edge band that holds a catalogue VGB is split at the first
  (last) edge of the fine grid it was coarsened from, so the guard is one fine
  band (layer_df/8, the GB guard width) and the sources sit in band 1. A grid
  with every source interior is returned bit-identical, so stores without the
  problem resume unchanged. A source that cannot be guarded raises.
  12mo: VGB grid 58 -> 59 bands, lowest edges 0.277778 / **0.295139** /
  0.416667 mHz. The new edge is `17 * layer_df / 8`, the same value as the GB
  grid's second edge.
* `scripts/fstat_proposal/extend_gb_band_floor.py --branch vgb`: the matching
  store migration. Per-band arrays get one leading band (fresh values;
  `band_temps` copies the old lowest band), so the old band 0's records become
  band 1 and every band above moves up by one. Leaves, chain, `d_h` / `h_h` are
  untouched (VGB f0 is fixed and assigned to a band by `searchsorted` at
  propose time). `cap_edges` is rewritten to the band grid (cap-free branch).
  Refuses: a grid that is not one edge inserted inside the lowest band, an
  already-split store, unknown band-sized datasets, a mid-iteration checkpoint
  or running backup copy next to the store.

The GB grid, the GB F-stat epoch caches and the eigen-table sidecar
(`*_eigen_tables.pkl`: per-leaf EMRI / MBH / SOBBH entries only) are not
affected.

Verified locally on a copy of the Nov 1 extract: dry run, apply, resume read +
VGB band check pass with VGB counts per band `[0, 8, 2, 2, ...]`; the real
`VGBSetup.init_band_structure` on the production WDM layer grid builds the same
lowest edges. Tests: `tests/test_vgb_edge_band_guard.py`,
`tests/test_extend_gb_band_floor.py`.

## Steps (on the cluster)

```bash
LAT=/shared/home/gfreedm1/projects/cd1l/lisa-analysis-tools
OUT=/shared/data/global_fit_output
RUN=$OUT/gf_prod_12mo_flagship
STORE=$RUN/gf_prod_12mo_testing.h5
BAK=$OUT/gf_prod_12mo_flagship_backup_2026-11-05_vgb
source /shared/home/gfreedm1/projects/cd1l/.venv/bin/activate
```

### 1. Nothing running on the store

```bash
squeue -u $USER -o "%.10i %.28j %.8T %.10M"     # no gf12mo_flagship
```

If it is still running, cancel it just after a row saves (`[SAVE] save_step` in
the job log), never during an F-stat refit.

### 2. Code

```bash
cd $LAT && git pull && git log -1 --oneline
python -m unittest tests.test_vgb_edge_band_guard tests.test_extend_gb_band_floor   # OK
```

### 3. Back up the store (~24 GB)

```bash
df -h $OUT
mkdir $BAK && cp $STORE $BAK/ && cmp $STORE $BAK/gf_prod_12mo_testing.h5 && echo "store backup identical"
```

### 4. Move the mid-iteration state aside

```bash
mkdir -p $RUN/_pre_vgb_split
mv $RUN/gf_prod_12mo_testing_running_backup_copy.h5 $RUN/_pre_vgb_split/ 2>/dev/null
mv $RUN/gf_prod_12mo_testing_midit_checkpoint.pkl   $RUN/_pre_vgb_split/ 2>/dev/null
ls $RUN/_pre_vgb_split
```

### 5. Split: dry run, then apply

```bash
cd $LAT
python scripts/fstat_proposal/extend_gb_band_floor.py $STORE --branch vgb --guard-from-gb-grid
```

Expect `old vgb grid: 58 bands, lowest edges [0.277778 0.416667 0.555556]`,
`new vgb grid: 59 bands, lowest edges [0.277778 0.295139 0.416667 0.555556]`,
`VGBs in the old lowest band per walker (cold rung): [8, 8, 8, 8]`,
22 per-band datasets, no `REFUSING`, no `BLOCKERS`. Then:

```bash
python scripts/fstat_proposal/extend_gb_band_floor.py $STORE --branch vgb --guard-from-gb-grid --apply
```

It ends with `resume check (vgb): OK; cold walker-0 VGB count per band, first 4:
[0, 8, ...]`. On any failure: `cp $BAK/gf_prod_12mo_testing.h5 $STORE` and stop.

### 6. Resubmit

```bash
cd $LAT/scripts/flagship && ./submit_gf_12mo.sh
```

### 7. Startup checks

```bash
L=$OUT/gf12mo_flagship_<jobid>.log
grep -m1 "VGB edge-band guard" $L          # "8 catalogue source(s) ... (58 -> 59 sub-bands ...)"
grep -m1 "RESUMING from existing backend" $L
grep -E "Cannot resume|band grid mismatch" $L   # nothing
```

If the resume is refused with a VGB band-grid mismatch, `run.py` has written
`$RUN/vgb_config_band_edges.npy`. Restore the backup, then redo step 5 with
`--edges-npy $RUN/vgb_config_band_edges.npy` in place of `--guard-from-gb-grid`.

### 8. First rows: are the 8 VGBs sampled?

After a few saved rows:

```bash
python - <<'EOF'
import h5py, numpy as np
f = h5py.File("/shared/data/global_fit_output/gf_prod_12mo_flagship/gf_prod_12mo_testing.h5", "r")
v = f["global_fit/sub_backend/vgb"]; it = int(f["global_fit"].attrs["iteration"]) - 1
print("vgb bands", len(v["band_edges"]) - 1, "| cold in-model proposals, bands 0-3, last row:",
      v["band_num_proposed"][it][:4, 0])
hh = v["h_h"][it]
print("leaves with finite h_h:", int(np.isfinite(hh).all(0).sum()), "of", hh.shape[1])
EOF
```

Expect band 1's proposal counter to grow like the other bands' and all 55
leaves to have a finite `h_h`. The monitor's VGB note ("8 of 55 carry no
<h|h>") should disappear.

## Interpreting the samples

Rows before the resume carry no information on these 8 VGBs. Treat their
first ~100 rows after it as burn-in. The other 47 VGBs and every other branch
are unaffected.

## 18mo / 24mo

Fresh launches build the guarded grid automatically. A resume of any other
existing store whose VGB grid has a source in an edge band is refused by
the band-grid check and needs the same split.
