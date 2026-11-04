# Runbook: resume the first 12mo run in `full_pe` with the lower GB floor

*2026-11-03. Resume the first 12-month run, `gf_prod_12mo_flagship`, **in
place**, in `full_pe` where its chains stopped, with the GB band floor lowered
from 0.556 to 0.278 mHz (`GB_MIN_FREQ` 5.5e-4 -> 2.5e-4; f0 prior from
0.295 mHz). The fresh v1.2.8dev restart is cancelled. The folder is backed up
whole first.*

## What changes, and nothing else

The 12mo launcher differs from the one the run last used (`9e13197f`) in
exactly five executed lines:

| Line | Why |
|---|---|
| `GB_MIN_FREQ=2.5e-4` | the change asked for |
| `GB_PE_RJ_FSTAT_RESIDUAL_SPLIT` default `1` | was typed on the launch line since job 851; now built in. `full_pe` then has only the live-residual F-stat move: the GB-free grid move (`rj_fstat_pe`) fits its grid once on entry, and on this grid that fit ran 6.9 h and then ran the GPU out of memory (job 850). It never completed in this run. |
| `STAGE_SKIP_NOISE=1` | keeps the composed stages equal to the stored ones. Without it, today's driver puts `noise_search` / `noise_vgb_search` ahead of `full_pe` (the parent pin store is gone) and the store refuses the resume. It changes nothing else. |
| `GF_MONITOR_TITLE`, `GF_MONITOR_VERSION=v1.2.8dev` | monitor page labels only |

Folder, job name and log names are the run's own (`gf_prod_12mo_flagship`,
`gf12mo_flagship`, `gf12mo_flagship_<jobid>.log`).

The store itself changes in one way: the GB per-band records go from 1232 to
1248 bands. Lowering the floor only adds 16 bands at the bottom (every old
edge keeps its value), so `scripts/fstat_proposal/extend_gb_band_floor.py`
moves every per-band record up 16 places unchanged and gives the 16 new bands
the values a fresh run would. Sources, chains, other branches, the recipe and
every scalar are untouched. The run resumes in `full_pe` at the next row
(every search stage is stored complete).

Before resuming, three files in the folder that hold the 1232-band state are
moved aside (they remain in the backup): the running backup copy, the
mid-iteration checkpoint, and the F-stat epoch caches. As a result `full_pe`
fits a fresh live-residual F-stat grid at its first F-stat draw rather than at
its next 150-row refit.

The code is the current fork `dev` (v1.2.8dev), not the cherry-picked state
the run last used on 2026-11-01. List the differences with
`git log --oneline backup/dev-cherry-pick-2026-11-01..dev -- src scripts/fstat_proposal`.

Verified 2026-11-03 on the local snapshot of this store (row 1129): the run's
own resume read and GB band/cap checks pass with no reset or discard warnings;
every old band is carried over exactly at +16 (182 dataset/row checks); all
17,615 cold-chain sources map to their old band + 16; the un-extended store
is refused by the same checks; the stored-recipe check refuses the
noise-stage recipe and accepts the 5-stage one. Tests:
`tests/test_extend_gb_band_floor.py`, `tests/test_flagship_launchers.py`.

## Steps (on the cluster)

```bash
LAT=/shared/home/gfreedm1/projects/cd1l/lisa-analysis-tools
OUT=/shared/data/global_fit_output
RUN=$OUT/gf_prod_12mo_flagship                       # resumed in place
BAK=$OUT/gf_prod_12mo_flagship_backup_2026-11-03     # whole-folder backup
REF=$OUT/gf_prod_12mo_v1.2.8dev                      # supplies the new band grid
source /shared/home/gfreedm1/projects/cd1l/.venv/bin/activate
```

### 1. Code

```bash
cd $LAT && git pull && git log -1 --oneline
python scripts/flagship/build.py --check
```

### 2. Cancel v1.2.8dev; make sure nothing is running on the 12mo store

```bash
squeue -u $USER -o "%.10i %.28j %.8T %.10M"
scancel <jobid of gf12mo_v1.2.8dev>
squeue -u $USER                     # neither gf12mo_v1.2.8dev nor gf12mo_flagship
```

### 3. Back up what the resume changes (~23 GB)

A whole-folder copy (182 GB on 2026-11-03) does not fit: `/shared` had 170 GB
free, shared by everyone. It is also unnecessary. The resume MODIFIES only
the store `gf_prod_12mo_testing.h5`. `warmstart/` may lose old self-refit
epochs to pruning. Everything else is left alone, only appended to (logs,
artifacts, new F-stat epochs), or moved within the folder in step 4. So back
up the store, `warmstart/` and the small files:

```bash
df -h $OUT                                          # keep >= ~60 GB free after this
mkdir $BAK
rsync -a \
  --exclude 'gb_fstat_fit/' --exclude '*_running_backup_copy.h5' \
  --exclude '*.bak*' --exclude '*.pre_rerung-*.h5' --exclude '*_extract.h5' \
  --exclude 'gf_prod_12mo_artifacts/' --exclude 'slurm_stdout_*.log' --exclude 'gpu_*.csv' \
  $RUN/ $BAK/
du -sh $BAK                                         # ~23 GB
cmp $RUN/gf_prod_12mo_testing.h5 $BAK/gf_prod_12mo_testing.h5 && echo "store backup identical"
df -h $OUT
```

To restore: copy `gf_prod_12mo_testing.h5` and `warmstart/` back from
`$BAK`, and move the step-4 items back out of `_pre_lowf_1232bands/`.

### 4. Check write access and move the 1232-band leftovers aside

The folder and store are owned by mlkatz1, so `chmod` on them fails for
anyone else. Check that you can write (group access):

```bash
ls -ld $RUN $RUN/gb_fstat_fit
test -w $RUN && test -w $RUN/gf_prod_12mo_testing.h5 && test -w $RUN/gb_fstat_fit && echo writable
```

No `writable`: ask Mike to `chmod -R g+w` the folder. Then:

```bash
mkdir $RUN/_pre_lowf_1232bands
mv $RUN/gf_prod_12mo_testing_running_backup_copy.h5 $RUN/_pre_lowf_1232bands/ 2>/dev/null
mv $RUN/gf_prod_12mo_testing_midit_checkpoint.pkl   $RUN/_pre_lowf_1232bands/ 2>/dev/null
mkdir $RUN/_pre_lowf_1232bands/gb_fstat_fit
mv $RUN/gb_fstat_fit/shared* $RUN/_pre_lowf_1232bands/gb_fstat_fit/ 2>/dev/null
ls $RUN/_pre_lowf_1232bands $RUN/_pre_lowf_1232bands/gb_fstat_fit
```

### 5. Extend the band grid: dry run, then apply

```bash
cd $LAT
python scripts/fstat_proposal/extend_gb_band_floor.py \
    $RUN/gf_prod_12mo_testing.h5 --edges-from $REF/gf_prod_12mo_testing.h5
```

Expect `old gb grid: 1232 bands [0.555556, 21.944444] mHz`, `new gb grid:
1248 bands [0.277778, 21.944444] mHz`, `+16 bands, +16 cap cells`, `outside:
0`, about 26 datasets to extend, no `REFUSING`, no `BLOCKERS`. Any refusal:
stop; nothing was written. Then:

```bash
python scripts/fstat_proposal/extend_gb_band_floor.py \
    $RUN/gf_prod_12mo_testing.h5 --edges-from $REF/gf_prod_12mo_testing.h5 --apply
```

It ends with `resume check: OK`. If it fails part-way, restore the store from
the backup (`cp $BAK/gf_prod_12mo_testing.h5 $RUN/`) and repeat this step.
The extension writes the new per-band datasets before deleting the old ones,
so the store grows by roughly the size of its GB band arrays (a few GB) while
it runs. HDF5 does not give that space back.

### 6. Check the stages the launcher will compose

```bash
python -m unittest tests.test_flagship_launchers.RecipeCompositionTest      # OK
```

The 12mo case composes `gb_search_seed, source_search, gb_search_1,
gb_search_2, full_pe` without a noise pin, and `full_pe` carries
`rj_fstat_pe_residual` and no `rj_fstat_pe`.

### 7. Launch

```bash
cd $LAT/scripts/flagship && ./submit_gf_12mo.sh
```

No extra variables on the launch line: both `STAGE_SKIP_NOISE=1` and
`GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=1` are now in the launcher.

### 8. Startup checks (first ~40 min)

```bash
L=$OUT/gf12mo_flagship_<jobid>.log
grep -E "^\[FRESH\]|^\[LADDER\]" $L                  # "already holds a run -- RESUMING it"; rungs 24 = 24
grep -m5 "\[combined\] stage" $L                     # the 5 stages, no noise_search
grep -m1 "stage       full_pe" $L                    # rj_fstat_pe_residual, no rj_fstat_pe
grep -m1 "The number of subbands is" $L              # 1248
grep -m1 "RESUMING from existing backend" $L         # .../gf_prod_12mo_flagship/... at stored iteration N
grep -m1 "V9-STAGE full_pe\] entering" $L
grep -m1 "shutoff valve is LIVE" $L                  # 4 walkers x 1248 bands
grep -E "Cannot resume|band grid mismatch|would be inserted|corrupted band" $L    # nothing
grep -c Traceback $L                                 # read any hit; a rejected MBH waveform proposal is harmless
```

`[V9-SEED] ... not found -- no noise pin` still appears and is expected.

### 9. First saves

```bash
grep "\[SAVE\] save_step" $L | tail -3
python scripts/diagnostics/gf_last_iteration.py $RUN
```

The stored row count rises from N, the GB count starts where the run stopped
(~4,400 per walker), the log-likelihood continues from its last value, and the
monitor's left dotted red line sits at ~0.28 mHz. The first
`rj_fstat_pe_residual` draw fits a fresh F-stat epoch on the 1248-band grid.
The 24 h limit applies as before: resubmit the same launcher.

## Interpreting the samples

- Rows up to N are `full_pe` on the old frequency range; rows after N on the
  new one. The 16 new bands start empty and fill gradually. Treat the rows
  after N as burn-in until the GB count below 0.556 mHz levels off.
- The noise model has been absorbing the unmodelled low-frequency binaries;
  expect `Sa_a` / galfor to drift as sources appear there.
- Four VGBs sit at 0.31-0.34 mHz, now inside the GB range: check that no GB
  leaf settles on them.

## 18mo / 24mo

The 18mo's parent is the 12mo entry (`gf_prod_12mo_flagship`); the 24mo's is
the 18mo. Both still use `GB_PE_RJ_FSTAT_RESIDUAL_SPLIT=0.9`.
